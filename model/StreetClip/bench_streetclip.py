#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
KM_THRESHOLDS = [1, 25, 200, 750, 2500]


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(256)
        return "git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark StreetCLIP on OSV5M test images.")
    p.add_argument("--dataset-dir", type=Path, default=Path("dataset/osv5m_test"))
    p.add_argument("--csv-path", type=Path, default=Path("dataset/osv5m_test/test_filtered.csv"))
    p.add_argument("--model-dir", type=Path, default=Path("StreetCLIP"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/streetclip_eval"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--text-batch-size", type=int, default=256)
    p.add_argument("--max-images", type=int, default=0, help="0 = all images")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_str(x: object) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


def iso2_to_name(code: str) -> str:
    code = (code or "").strip().upper()
    if not code:
        return code
    try:
        import pycountry  # type: ignore

        c = pycountry.countries.get(alpha_2=code)
        if c is not None and getattr(c, "name", None):
            return c.name
    except Exception:
        pass
    return code


def pick_city(row: pd.Series) -> Optional[str]:
    city = clean_str(row.get("city"))
    if city:
        return city
    unique_city = clean_str(row.get("unique_city"))
    if unique_city:
        return unique_city.split("_")[0].strip() or None
    return None


def build_image_map(dataset_dir: Path) -> Tuple[Dict[str, str], int]:
    id_to_path: Dict[str, str] = {}
    duplicate_count = 0
    for p in dataset_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        img_id = p.stem
        if img_id in id_to_path:
            duplicate_count += 1
            continue
        id_to_path[img_id] = str(p)
    return id_to_path, duplicate_count


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    r = 6371.0088
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1 - a, 1e-12, 1.0)))
    return r * c


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        return img, idx


def collate_pil(batch):
    images = [x[0] for x in batch]
    idxs = [x[1] for x in batch]
    return images, torch.tensor(idxs, dtype=torch.long)


def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    labels: List[str],
    device: torch.device,
    text_batch_size: int,
) -> torch.Tensor:
    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(labels), text_batch_size):
            chunk = labels[i : i + text_batch_size]
            tokenized = processor(text=chunk, return_tensors="pt", padding=True, truncation=True)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                txt = model.get_text_features(**tokenized)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            feats.append(txt.detach().cpu())
    return torch.cat(feats, dim=0)


@dataclass
class EvalStats:
    n: int = 0
    country_correct: int = 0
    city_hier_den: int = 0
    city_hier_correct: int = 0
    city_oracle_den: int = 0
    city_oracle_correct: int = 0


def pct_at_km(dist: np.ndarray, km: int) -> float:
    if dist.size == 0:
        return float("nan")
    return float((dist <= km).mean() * 100.0)


def summarize_distance_metrics(dist_km: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {
        "mean_km": float(np.mean(dist_km)) if dist_km.size else float("nan"),
        "median_km": float(np.median(dist_km)) if dist_km.size else float("nan"),
    }
    for k in KM_THRESHOLDS:
        out[f"pct_at_{k}km"] = pct_at_km(dist_km, k)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"CWD={os.getcwd()}")
    print(f"dataset_dir={args.dataset_dir.resolve()}")
    print(f"csv_path={args.csv_path.resolve()}")
    print(f"model_dir={args.model_dir.resolve()}")

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {args.dataset_dir}")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"csv_path not found: {args.csv_path}")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {args.model_dir}")

    print("Loading CSV...")
    usecols = ["id", "latitude", "longitude", "country", "city", "unique_city"]
    df = pd.read_csv(args.csv_path, usecols=usecols)
    df["id"] = df["id"].astype(str)
    df["country_code"] = df["country"].map(lambda x: clean_str(x) or "")
    df["country_name"] = df["country_code"].map(iso2_to_name)
    df["city_name"] = df.apply(pick_city, axis=1)
    print(f"CSV rows={len(df)}")

    print("Indexing images...")
    id_to_path, duplicate_images = build_image_map(args.dataset_dir)
    print(f"Indexed images={len(id_to_path)} (duplicates skipped={duplicate_images})")

    df["image_path"] = df["id"].map(id_to_path)
    before = len(df)
    df = df.dropna(subset=["image_path"]).copy()
    print(f"Rows with existing image={len(df)} (dropped={before - len(df)})")

    if args.max_images > 0 and len(df) > args.max_images:
        df = df.sample(n=args.max_images, random_state=args.seed).copy()
        print(f"Subsample active: {len(df)} images")

    df = df.reset_index(drop=True)

    city_df = df.dropna(subset=["city_name"]).copy()
    city_centroids_df = (
        city_df.groupby(["country_code", "city_name"], as_index=False)[["latitude", "longitude"]]
        .median()
        .rename(columns={"latitude": "city_lat", "longitude": "city_lon"})
    )
    city_coord: Dict[Tuple[str, str], Tuple[float, float]] = {
        (r["country_code"], r["city_name"]): (float(r["city_lat"]), float(r["city_lon"]))
        for _, r in city_centroids_df.iterrows()
    }
    country_centroids_df = (
        df.groupby(["country_code", "country_name"], as_index=False)[["latitude", "longitude"]]
        .median()
        .rename(columns={"latitude": "country_lat", "longitude": "country_lon"})
    )
    country_coord: Dict[str, Tuple[float, float]] = {
        r["country_code"]: (float(r["country_lat"]), float(r["country_lon"]))
        for _, r in country_centroids_df.iterrows()
    }

    countries = country_centroids_df["country_code"].tolist()
    country_name_by_code = dict(
        zip(country_centroids_df["country_code"], country_centroids_df["country_name"])
    )
    country_to_idx = {c: i for i, c in enumerate(countries)}

    cities_by_country: Dict[str, List[str]] = {}
    for c, g in city_centroids_df.groupby("country_code"):
        cities_by_country[c] = sorted(g["city_name"].tolist())

    print(f"Countries in eval set={len(countries)}")
    print(f"Country-city pairs={len(city_centroids_df)}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device={device}")
    if str(device) != "cuda":
        print("WARNING: CUDA not available, this will be slower.")

    print("Loading StreetCLIP...")
    weights_path = args.model_dir / "pytorch_model.bin"
    if is_git_lfs_pointer(weights_path):
        raise RuntimeError(
            "StreetCLIP/pytorch_model.bin is a Git LFS pointer, not real weights. "
            "Fetch real weights first (git-lfs pull or Hugging Face snapshot)."
        )
    model = CLIPModel.from_pretrained(str(args.model_dir), local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    model.eval()

    print("Encoding country label texts...")
    country_prompts = [f"a street view photo in {country_name_by_code[c]}" for c in countries]
    country_text_feats = encode_texts(
        model=model,
        processor=processor,
        labels=country_prompts,
        device=device,
        text_batch_size=args.text_batch_size,
    ).to(device)

    print("Encoding city label texts per country...")
    city_label_cache: Dict[str, List[str]] = {}
    city_lat_cache: Dict[str, np.ndarray] = {}
    city_lon_cache: Dict[str, np.ndarray] = {}
    city_feat_cache_cpu: Dict[str, torch.Tensor] = {}
    for c in tqdm(countries, desc="Text(city)"):
        city_labels = cities_by_country.get(c, [])
        city_label_cache[c] = city_labels
        if not city_labels:
            continue
        city_lat_cache[c] = np.array([city_coord[(c, city)][0] for city in city_labels], dtype=np.float64)
        city_lon_cache[c] = np.array([city_coord[(c, city)][1] for city in city_labels], dtype=np.float64)
        prompts = [f"a street view photo in {city}, {country_name_by_code[c]}" for city in city_labels]
        feats = encode_texts(
            model=model,
            processor=processor,
            labels=prompts,
            device=device,
            text_batch_size=args.text_batch_size,
        )
        city_feat_cache_cpu[c] = feats

    image_paths = df["image_path"].tolist()
    eval_ds = ImagePathDataset(image_paths=image_paths)
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
        pin_memory=(device.type == "cuda"),
    )

    stats = EvalStats()
    dist_hier_all: List[float] = []
    dist_oracle_all: List[float] = []

    city_feat_cache_gpu: Dict[str, torch.Tensor] = {}

    print("Running evaluation...")
    with torch.no_grad():
        for pil_images, local_idxs in tqdm(loader, desc="Eval"):
            batch_df = df.iloc[local_idxs.numpy()]
            inputs = processor(images=pil_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                image_feats = model.get_image_features(pixel_values=pixel_values)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

            country_logits = image_feats @ country_text_feats.T
            pred_country_idx = torch.argmax(country_logits, dim=1).cpu().numpy()
            pred_country_codes = [countries[i] for i in pred_country_idx]

            gt_country_codes = batch_df["country_code"].tolist()
            gt_city_names = batch_df["city_name"].tolist()
            gt_lat = batch_df["latitude"].to_numpy(dtype=np.float64)
            gt_lon = batch_df["longitude"].to_numpy(dtype=np.float64)

            stats.n += len(batch_df)
            stats.country_correct += sum(
                1 for g, p in zip(gt_country_codes, pred_country_codes) if g == p
            )

            bs = len(batch_df)
            pred_lat_h = np.array([country_coord[c][0] for c in pred_country_codes], dtype=np.float64)
            pred_lon_h = np.array([country_coord[c][1] for c in pred_country_codes], dtype=np.float64)
            pred_lat_o = np.array([country_coord[c][0] for c in gt_country_codes], dtype=np.float64)
            pred_lon_o = np.array([country_coord[c][1] for c in gt_country_codes], dtype=np.float64)
            pred_city_h: List[Optional[str]] = [None] * bs
            pred_city_o: List[Optional[str]] = [None] * bs

            pred_country_arr = np.array(pred_country_codes, dtype=object)
            for c in np.unique(pred_country_arr):
                if c not in city_feat_cache_cpu or len(city_label_cache[c]) == 0:
                    continue
                idxs = np.where(pred_country_arr == c)[0]
                if c not in city_feat_cache_gpu:
                    city_feat_cache_gpu[c] = city_feat_cache_cpu[c].to(device)
                logits_city = image_feats[idxs] @ city_feat_cache_gpu[c].T
                top = torch.argmax(logits_city, dim=1).cpu().numpy()
                pred_lat_h[idxs] = city_lat_cache[c][top]
                pred_lon_h[idxs] = city_lon_cache[c][top]
                labels = city_label_cache[c]
                for rel, global_idx in enumerate(idxs):
                    pred_city_h[global_idx] = labels[int(top[rel])]

            gt_country_arr = np.array(gt_country_codes, dtype=object)
            for c in np.unique(gt_country_arr):
                if c not in city_feat_cache_cpu or len(city_label_cache[c]) == 0:
                    continue
                idxs = np.where(gt_country_arr == c)[0]
                if c not in city_feat_cache_gpu:
                    city_feat_cache_gpu[c] = city_feat_cache_cpu[c].to(device)
                logits_city_o = image_feats[idxs] @ city_feat_cache_gpu[c].T
                top_o = torch.argmax(logits_city_o, dim=1).cpu().numpy()
                pred_lat_o[idxs] = city_lat_cache[c][top_o]
                pred_lon_o[idxs] = city_lon_cache[c][top_o]
                labels = city_label_cache[c]
                for rel, global_idx in enumerate(idxs):
                    pred_city_o[global_idx] = labels[int(top_o[rel])]

            for i in range(bs):
                gt_city = gt_city_names[i]
                if gt_city is None:
                    continue
                if pred_city_h[i] is not None:
                    stats.city_hier_den += 1
                    stats.city_hier_correct += int(
                        (pred_country_codes[i] == gt_country_codes[i]) and (pred_city_h[i] == gt_city)
                    )
                if pred_city_o[i] is not None:
                    stats.city_oracle_den += 1
                    stats.city_oracle_correct += int(pred_city_o[i] == gt_city)

            dist_h = haversine_km(gt_lat, gt_lon, pred_lat_h, pred_lon_h)
            dist_o = haversine_km(gt_lat, gt_lon, pred_lat_o, pred_lon_o)
            dist_hier_all.extend(dist_h.tolist())
            dist_oracle_all.extend(dist_o.tolist())

    dist_hier_np = np.array(dist_hier_all, dtype=np.float64)
    dist_oracle_np = np.array(dist_oracle_all, dtype=np.float64)

    summary = {
        "n_images": int(stats.n),
        "country_acc_top1": float(stats.country_correct / stats.n) if stats.n else float("nan"),
        "city_acc_top1_hierarchical": float(stats.city_hier_correct / stats.city_hier_den)
        if stats.city_hier_den
        else float("nan"),
        "city_acc_top1_oracle_country": float(stats.city_oracle_correct / stats.city_oracle_den)
        if stats.city_oracle_den
        else float("nan"),
        "distance_km_hierarchical": summarize_distance_metrics(dist_hier_np),
        "distance_km_oracle_country": summarize_distance_metrics(dist_oracle_np),
    }

    print("\n===== STREETCLIP EVAL SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_json = args.output_dir / "streetclip_eval_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
