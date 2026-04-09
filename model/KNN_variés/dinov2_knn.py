from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


DATASET_DIR = Path("dataset_OSV5M/datasets/osv5m")
CSV_PATH = DATASET_DIR / "/metadata_filtered/rest_filtered_v2.csv"
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/dinov2_knn_geo.pt")
DEFAULT_OUTPUT_PATH = Path("checkpoints/dinov2_knn_geo_index.pt")
DEFAULT_SUMMARY_PATH = Path("checkpoints/dinov2_knn_geo_summary.json")
DEFAULT_DINOV2_MODEL = "facebook/dinov2-large"
SELECTION_METRIC = "country_penalized_mean_km"


@dataclass(frozen=True)
class GeoSample:
    image_path: str
    image_id: str
    country: str
    latitude: float
    longitude: float
    target_vec: torch.Tensor


class GeoImageDataset(Dataset):
    def __init__(self, samples: Sequence[GeoSample], transform: transforms.Compose):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        image = self.transform(image)
        latlon = torch.tensor([sample.latitude, sample.longitude], dtype=torch.float32)
        return image, sample.target_vec.clone(), latlon, sample.image_id, sample.country


class DinoV2FeatureExtractor(nn.Module):
    def __init__(self, model_name_or_path: str, local_files_only: bool = False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.feature_dim = int(self.model.config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return normalize_vec(pooled.float())


class GeoKNNRegressor:
    def __init__(self, temperature: float = 0.05, ref_chunk_size: int = 8192):
        self.temperature = temperature
        self.ref_chunk_size = ref_chunk_size
        self.bank_features: torch.Tensor | None = None
        self.bank_targets: torch.Tensor | None = None
        self.bank_ids: list[str] = []
        self.bank_countries: list[str] = []

    def fit(self, features: torch.Tensor, targets: torch.Tensor, ids: list[str], countries: list[str]) -> None:
        if features.ndim != 2:
            raise ValueError("features must be a 2D tensor")
        if len(features) == 0:
            raise ValueError("cannot fit KNN with an empty feature bank")
        self.bank_features = normalize_vec(features.float().cpu())
        self.bank_targets = normalize_vec(targets.float().cpu())
        self.bank_ids = list(ids)
        self.bank_countries = list(countries)

    @torch.no_grad()
    def predict_from_features(
        self,
        query_features: torch.Tensor,
        k: int,
        query_chunk_size: int = 256,
    ) -> tuple[torch.Tensor, list[list[str]], torch.Tensor, list[str]]:
        if self.bank_features is None or self.bank_targets is None:
            raise RuntimeError("the KNN regressor must be fitted before prediction")

        bank_features = self.bank_features
        bank_targets = self.bank_targets
        k = max(1, min(k, bank_features.size(0)))
        query_features = normalize_vec(query_features.float().cpu())

        predictions: list[torch.Tensor] = []
        neighbor_ids: list[list[str]] = []
        neighbor_scores: list[torch.Tensor] = []
        predicted_countries: list[str] = []

        for start in range(0, query_features.size(0), query_chunk_size):
            query_chunk = query_features[start : start + query_chunk_size]
            best_scores = torch.full((query_chunk.size(0), k), -1e9, dtype=torch.float32)
            best_indices = torch.zeros((query_chunk.size(0), k), dtype=torch.long)

            for ref_start in range(0, bank_features.size(0), self.ref_chunk_size):
                ref_chunk = bank_features[ref_start : ref_start + self.ref_chunk_size]
                similarities = query_chunk @ ref_chunk.T
                chunk_k = min(k, ref_chunk.size(0))
                chunk_scores, chunk_indices = similarities.topk(chunk_k, dim=1)
                chunk_indices = chunk_indices + ref_start

                merged_scores = torch.cat([best_scores, chunk_scores], dim=1)
                merged_indices = torch.cat([best_indices, chunk_indices], dim=1)
                best_scores, best_positions = merged_scores.topk(k, dim=1)
                best_indices = torch.gather(merged_indices, 1, best_positions)

            current_targets = bank_targets[best_indices]
            raw_weights = torch.softmax(best_scores / max(self.temperature, 1e-3), dim=1)
            country_mask = torch.zeros_like(raw_weights)
            chunk_predicted_countries: list[str] = []

            for row_idx, row_indices in enumerate(best_indices.tolist()):
                country_to_weight: dict[str, float] = {}
                for col_idx, bank_index in enumerate(row_indices):
                    country = self.bank_countries[bank_index]
                    country_to_weight[country] = country_to_weight.get(country, 0.0) + float(raw_weights[row_idx, col_idx])

                winner_country = max(country_to_weight.items(), key=lambda item: item[1])[0]
                chunk_predicted_countries.append(winner_country)
                for col_idx, bank_index in enumerate(row_indices):
                    if self.bank_countries[bank_index] == winner_country:
                        country_mask[row_idx, col_idx] = 1.0

            filtered_weights = raw_weights * country_mask
            filtered_weights = filtered_weights / filtered_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            pred_vec = normalize_vec((filtered_weights.unsqueeze(-1) * current_targets).sum(dim=1))

            predictions.append(pred_vec)
            neighbor_scores.append(best_scores)
            predicted_countries.extend(chunk_predicted_countries)
            for row in best_indices.tolist():
                neighbor_ids.append([self.bank_ids[index] for index in row])

        return torch.cat(predictions, dim=0), neighbor_ids, torch.cat(neighbor_scores, dim=0), predicted_countries


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)


def normalize_vec(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


def latlon_to_unitvec(lat_deg, lon_deg) -> torch.Tensor:
    lat = torch.deg2rad(torch.tensor(lat_deg, dtype=torch.float32))
    lon = torch.deg2rad(torch.tensor(lon_deg, dtype=torch.float32))
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def unitvec_to_latlon(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v = normalize_vec(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    lat = torch.asin(z)
    lon = torch.atan2(y, x)
    return torch.rad2deg(lat), torch.rad2deg(lon)


def haversine_km(lat1, lon1, lat2, lon2) -> torch.Tensor:
    radius_km = 6371.0
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.clamp(a.sqrt(), 0.0, 1.0))
    return radius_km * c


def parse_k_values(raw_value: str) -> list[int]:
    values = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("at least one k value is required")
    return sorted(set(values))


def make_square_size(size_config) -> int:
    if isinstance(size_config, dict):
        if "height" in size_config:
            return int(size_config["height"])
        if "shortest_edge" in size_config:
            return int(size_config["shortest_edge"])
    if isinstance(size_config, (list, tuple)):
        return int(size_config[0])
    return int(size_config)


def build_transforms(
    image_size: int,
    image_mean: list[float],
    image_std: list[float],
    num_augmented_views: int,
    rotation_deg: float,
    color_jitter: float,
    enable_hflip: bool,
) -> tuple[transforms.Compose, transforms.Compose]:
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )

    if num_augmented_views <= 0:
        return eval_transform, eval_transform

    steps: list[object] = [
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05),
        ),
    ]
    if rotation_deg > 0:
        steps.append(transforms.RandomRotation(degrees=rotation_deg))
    if enable_hflip:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if color_jitter > 0:
        steps.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(0.5, color_jitter / 2),
            )
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )
    return eval_transform, transforms.Compose(steps)


def load_geo_samples(dataset_dir: Path, csv_path: Path) -> list[GeoSample]:
    df = pd.read_csv(csv_path)
    df["id"] = df["id"].astype(str)

    target_vectors = latlon_to_unitvec(df["latitude"].values, df["longitude"].values)
    id_to_row: dict[str, tuple[str, float, float, torch.Tensor]] = {}
    for row, target_vec in zip(df.itertuples(index=False), target_vectors):
        country = str(row.country) if pd.notna(row.country) else "UNK"
        id_to_row[str(row.id)] = (country, float(row.latitude), float(row.longitude), target_vec)

    image_folder = datasets.ImageFolder(root=str(dataset_dir))
    matched_samples: list[GeoSample] = []
    missing_in_csv = 0

    for image_path, _ in image_folder.samples:
        image_id = Path(image_path).stem
        geo_data = id_to_row.get(image_id)
        if geo_data is None:
            missing_in_csv += 1
            continue
        country, latitude, longitude, target_vec = geo_data
        matched_samples.append(
            GeoSample(
                image_path=image_path,
                image_id=image_id,
                country=country,
                latitude=latitude,
                longitude=longitude,
                target_vec=target_vec,
            )
        )

    if not matched_samples:
        raise RuntimeError("no image could be matched with the CSV coordinates")

    print(f"CSV rows: {len(df)}")
    print(f"Matched images: {len(matched_samples)}")
    print(f"Images skipped because their id is missing in CSV: {missing_in_csv}")
    return matched_samples


def split_samples(samples: Sequence[GeoSample], seed: int) -> tuple[list[GeoSample], list[GeoSample], list[GeoSample]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(samples), generator=generator).tolist()
    shuffled = [samples[idx] for idx in permutation]

    n_total = len(shuffled)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train : n_train + n_val]
    test_samples = shuffled[n_train + n_val :]
    return train_samples, val_samples, test_samples


def maybe_limit(samples: Sequence[GeoSample], max_samples: int | None) -> list[GeoSample]:
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return list(samples)
    return list(samples[:max_samples])


def build_loader(
    samples: Sequence[GeoSample],
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    dataset = GeoImageDataset(samples=samples, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    model.eval()
    feature_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    latlon_chunks: list[torch.Tensor] = []
    image_ids: list[str] = []
    countries: list[str] = []

    for images, targets, latlon, batch_ids, batch_countries in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        features = model(images).cpu()
        feature_chunks.append(features)
        target_chunks.append(targets.float().cpu())
        latlon_chunks.append(latlon.float().cpu())
        image_ids.extend(list(batch_ids))
        countries.extend(list(batch_countries))

    if not feature_chunks:
        raise RuntimeError(f"no features were extracted for: {description}")

    return (
        torch.cat(feature_chunks, dim=0),
        torch.cat(target_chunks, dim=0),
        torch.cat(latlon_chunks, dim=0),
        image_ids,
        countries,
    )


def append_to_bank(
    bank_features: torch.Tensor | None,
    bank_targets: torch.Tensor | None,
    bank_ids: list[str],
    bank_countries: list[str],
    new_features: torch.Tensor,
    new_targets: torch.Tensor,
    new_ids: list[str],
    new_countries: list[str],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    if bank_features is None or bank_targets is None:
        bank_features = new_features.float().cpu()
        bank_targets = new_targets.float().cpu()
    else:
        bank_features = torch.cat([bank_features, new_features.float().cpu()], dim=0)
        bank_targets = torch.cat([bank_targets, new_targets.float().cpu()], dim=0)
    bank_ids = bank_ids + list(new_ids)
    bank_countries = bank_countries + list(new_countries)
    return bank_features, bank_targets, bank_ids, bank_countries


def compute_metrics(
    pred_vec: torch.Tensor,
    true_latlon: torch.Tensor,
    pred_countries: list[str],
    true_countries: list[str],
    country_penalty_multiplier: float,
) -> dict[str, float]:
    pred_lat, pred_lon = unitvec_to_latlon(pred_vec)
    true_lat = true_latlon[:, 0]
    true_lon = true_latlon[:, 1]
    distances = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
    wrong_country = torch.tensor(
        [pred_country != true_country for pred_country, true_country in zip(pred_countries, true_countries)],
        dtype=torch.float32,
    )
    penalized_distances = distances * (1.0 + wrong_country * max(0.0, country_penalty_multiplier - 1.0))

    return {
        "mean_km": float(distances.mean().item()),
        "median_km": float(distances.median().item()),
        "country_penalized_mean_km": float(penalized_distances.mean().item()),
        "acc_at_1km": float((distances <= 1.0).float().mean().item()),
        "acc_at_25km": float((distances <= 25.0).float().mean().item()),
        "acc_at_200km": float((distances <= 200.0).float().mean().item()),
        "country_acc": float((1.0 - wrong_country).mean().item()),
        "wrong_country_rate": float(wrong_country.mean().item()),
    }


@torch.no_grad()
def evaluate_knn(
    knn: GeoKNNRegressor,
    query_features: torch.Tensor,
    query_latlon: torch.Tensor,
    query_countries: list[str],
    k: int,
    query_chunk_size: int,
    country_penalty_multiplier: float,
) -> dict[str, float]:
    pred_vec, _, _, pred_countries = knn.predict_from_features(
        query_features,
        k=k,
        query_chunk_size=query_chunk_size,
    )
    metrics = compute_metrics(
        pred_vec,
        query_latlon,
        pred_countries=pred_countries,
        true_countries=query_countries,
        country_penalty_multiplier=country_penalty_multiplier,
    )
    metrics["k"] = int(k)
    return metrics


def checkpoint_run_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_dir": str(args.dataset_dir),
        "csv_path": str(args.csv_path),
        "dinov2_model_name": args.dinov2_model_name,
        "dinov2_local_files_only": args.dinov2_local_files_only,
        "image_size": args.image_size,
        "num_augmented_views": args.num_augmented_views,
        "rotation_deg": args.rotation_deg,
        "color_jitter": args.color_jitter,
        "horizontal_flip": args.horizontal_flip,
        "seed": args.seed,
        "country_penalty_multiplier": args.country_penalty_multiplier,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
    }


def maybe_load_checkpoint(args: argparse.Namespace) -> dict | None:
    checkpoint_path = Path(args.checkpoint_path)
    if not args.resume:
        return None
    if not checkpoint_path.exists():
        print(f"[RESUME] no checkpoint found at {checkpoint_path.resolve()}, starting from scratch")
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_config = checkpoint.get("run_config")
    current_config = checkpoint_run_config(args)
    if saved_config is not None and saved_config != current_config:
        raise ValueError(
            "checkpoint configuration does not match the current run. "
            "Use the same data/augmentation settings or launch with --no-resume."
        )

    print(
        f"[RESUME] {checkpoint_path} | stage={checkpoint.get('stage', 'unknown')} | "
        f"next_aug_index={checkpoint.get('next_aug_index', 0)}"
    )
    return checkpoint


def save_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    stage: str,
    bank_features: torch.Tensor | None,
    bank_targets: torch.Tensor | None,
    bank_ids: list[str],
    bank_countries: list[str],
    train_clean_done: bool,
    next_aug_index: int,
    val_features: torch.Tensor | None = None,
    val_latlon: torch.Tensor | None = None,
    val_countries: list[str] | None = None,
    best_k: int | None = None,
    best_metrics: dict[str, float] | None = None,
    test_metrics: dict[str, float] | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "run_config": checkpoint_run_config(args),
            "train_clean_done": train_clean_done,
            "next_aug_index": next_aug_index,
            "bank_features": None if bank_features is None else bank_features.half(),
            "bank_targets": None if bank_targets is None else bank_targets.half(),
            "bank_ids": bank_ids,
            "bank_countries": bank_countries,
            "val_features": None if val_features is None else val_features.half(),
            "val_latlon": val_latlon,
            "val_countries": val_countries,
            "best_k": best_k,
            "best_metrics": best_metrics,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )


def save_outputs(
    output_path: Path,
    summary_path: Path,
    knn: GeoKNNRegressor,
    best_k: int,
    args: argparse.Namespace,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float] | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if knn.bank_features is None or knn.bank_targets is None:
        raise RuntimeError("cannot save an unfitted KNN model")

    torch.save(
        {
            "dinov2_model_name": args.dinov2_model_name,
            "image_size": args.image_size,
            "temperature": args.temperature,
            "country_penalty_multiplier": args.country_penalty_multiplier,
            "best_k": best_k,
            "bank_features": knn.bank_features.half(),
            "bank_targets": knn.bank_targets.half(),
            "bank_ids": knn.bank_ids,
            "bank_countries": knn.bank_countries,
        },
        output_path,
    )

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "output_path": str(output_path),
        "dinov2_model_name": args.dinov2_model_name,
        "dinov2_local_files_only": args.dinov2_local_files_only,
        "image_size": args.image_size,
        "num_augmented_views": args.num_augmented_views,
        "rotation_deg": args.rotation_deg,
        "color_jitter": args.color_jitter,
        "horizontal_flip": args.horizontal_flip,
        "temperature": args.temperature,
        "country_penalty_multiplier": args.country_penalty_multiplier,
        "selection_metric": SELECTION_METRIC,
        "k_values": args.k_values,
        "best_k": best_k,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DINOv2 + KNN geolocation model with checkpoint/resume support.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--dinov2-model-name", type=str, default=DEFAULT_DINOV2_MODEL)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-augmented-views", type=int, default=0)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--color-jitter", type=float, default=0.0)
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--country-penalty-multiplier", type=float, default=10.0)
    parser.add_argument("--k-values", type=parse_k_values, default=parse_k_values("1,2,3"))
    parser.add_argument("--ref-chunk-size", type=int, default=8192)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--us-max-samples", type=int, default=5000, help="Max US samples in training bank (undersampling)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    checkpoint_path = Path(args.checkpoint_path)

    print(f"device: {device}")
    print(f"dataset_dir: {args.dataset_dir.resolve()}")
    print(f"csv_path: {args.csv_path.resolve()}")
    print(f"checkpoint_path: {checkpoint_path.resolve()}")
    print(f"dinov2_model_name: {args.dinov2_model_name}")
    print(f"selection_metric: {SELECTION_METRIC}")

    image_processor = AutoImageProcessor.from_pretrained(
        args.dinov2_model_name,
        local_files_only=args.dinov2_local_files_only,
    )
    image_size = args.image_size if args.image_size > 0 else make_square_size(image_processor.size)
    image_mean = list(getattr(image_processor, "image_mean", [0.485, 0.456, 0.406]))
    image_std = list(getattr(image_processor, "image_std", [0.229, 0.224, 0.225]))
    eval_transform, train_transform = build_transforms(
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        num_augmented_views=args.num_augmented_views,
        rotation_deg=args.rotation_deg,
        color_jitter=args.color_jitter,
        enable_hflip=args.horizontal_flip,
    )

    all_samples = load_geo_samples(args.dataset_dir, args.csv_path)
    train_samples, val_samples, test_samples = split_samples(all_samples, seed=args.seed)

    train_samples = maybe_limit(train_samples, args.max_train_samples)
    val_samples = maybe_limit(val_samples, args.max_val_samples)
    test_samples = maybe_limit(test_samples, args.max_test_samples)
    
    # ── Undersampling US ──────────────────────────────────────────────
    US_MAX =  args.us_max_samples

    us_samples    = [s for s in train_samples if s.country == "US"]
    non_us_samples = [s for s in train_samples if s.country != "US"]

    if len(us_samples) > US_MAX:
        rng = random.Random(args.seed)
        us_samples = rng.sample(us_samples, US_MAX)
        print(f"Undersampling US : {len(us_samples)} / {len([s for s in train_samples if s.country == 'US'])} gardés")

    train_samples = non_us_samples + us_samples
    rng2 = random.Random(args.seed + 1)
    rng2.shuffle(train_samples)

    print(f"Train après undersampling : {len(train_samples)} samples")
    # ─────────────────────────────────────────────────────────────────

    if not train_samples:
        raise RuntimeError("the training split is empty after filtering/limiting")
    if not val_samples:
        raise RuntimeError("the validation split is empty after filtering/limiting")

    print(f"Train split: {len(train_samples)}")
    print(f"Val split: {len(val_samples)}")
    print(f"Test split: {len(test_samples)}")
    print(f"Effective image size: {image_size}")

    clean_train_loader = build_loader(train_samples, eval_transform, args.batch_size, args.num_workers, args.seed)
    val_loader = build_loader(val_samples, eval_transform, args.batch_size, args.num_workers, args.seed + 100)
    test_loader = build_loader(test_samples, eval_transform, args.batch_size, args.num_workers, args.seed + 200)

    feature_extractor = DinoV2FeatureExtractor(
        model_name_or_path=args.dinov2_model_name,
        local_files_only=args.dinov2_local_files_only,
    ).to(device)
    feature_extractor.eval()
    print(f"Embedding dimension: {feature_extractor.feature_dim}")

    checkpoint = maybe_load_checkpoint(args)
    train_clean_done = False
    next_aug_index = 0
    bank_features: torch.Tensor | None = None
    bank_targets: torch.Tensor | None = None
    bank_ids: list[str] = []
    bank_countries: list[str] = []
    val_features: torch.Tensor | None = None
    val_latlon: torch.Tensor | None = None
    val_countries: list[str] | None = None

    if checkpoint is not None:
        train_clean_done = bool(checkpoint.get("train_clean_done", False))
        next_aug_index = int(checkpoint.get("next_aug_index", 0))
        saved_bank_features = checkpoint.get("bank_features")
        saved_bank_targets = checkpoint.get("bank_targets")
        bank_features = None if saved_bank_features is None else saved_bank_features.float()
        bank_targets = None if saved_bank_targets is None else saved_bank_targets.float()
        bank_ids = list(checkpoint.get("bank_ids", []))
        bank_countries = list(checkpoint.get("bank_countries", []))
        saved_val_features = checkpoint.get("val_features")
        val_features = None if saved_val_features is None else saved_val_features.float()
        val_latlon = checkpoint.get("val_latlon")
        loaded_val_countries = checkpoint.get("val_countries")
        val_countries = None if loaded_val_countries is None else list(loaded_val_countries)

    def augment_loader_factory(aug_index: int) -> DataLoader:
        return build_loader(train_samples, train_transform, args.batch_size, args.num_workers, args.seed + 1000 + aug_index)

    if not train_clean_done:
        clean_features, clean_targets, _, clean_ids, clean_countries = extract_features(
            feature_extractor,
            clean_train_loader,
            device=device,
            description="Train clean views",
        )
        bank_features, bank_targets, bank_ids, bank_countries = append_to_bank(
            bank_features,
            bank_targets,
            bank_ids,
            bank_countries,
            clean_features,
            clean_targets,
            clean_ids,
            clean_countries,
        )
        train_clean_done = True
        save_checkpoint(
            checkpoint_path,
            args,
            stage="train_clean_ready",
            bank_features=bank_features,
            bank_targets=bank_targets,
            bank_ids=bank_ids,
            bank_countries=bank_countries,
            train_clean_done=True,
            next_aug_index=next_aug_index,
            val_features=val_features,
            val_latlon=val_latlon,
            val_countries=val_countries,
        )

    for aug_index in range(next_aug_index, max(0, args.num_augmented_views)):
        aug_loader = augment_loader_factory(aug_index)
        aug_features, aug_targets, _, aug_ids, aug_countries = extract_features(
            feature_extractor,
            aug_loader,
            device=device,
            description=f"Train augmented views {aug_index + 1}/{args.num_augmented_views}",
        )
        bank_features, bank_targets, bank_ids, bank_countries = append_to_bank(
            bank_features,
            bank_targets,
            bank_ids,
            bank_countries,
            aug_features,
            aug_targets,
            aug_ids,
            aug_countries,
        )
        next_aug_index = aug_index + 1
        save_checkpoint(
            checkpoint_path,
            args,
            stage=f"train_aug_{next_aug_index}_ready",
            bank_features=bank_features,
            bank_targets=bank_targets,
            bank_ids=bank_ids,
            bank_countries=bank_countries,
            train_clean_done=True,
            next_aug_index=next_aug_index,
            val_features=val_features,
            val_latlon=val_latlon,
            val_countries=val_countries,
        )

    if bank_features is None or bank_targets is None:
        raise RuntimeError("failed to build the KNN reference bank")

    print(f"Reference bank size after augmentation: {len(bank_features)}")

    if val_features is None or val_latlon is None or val_countries is None:
        val_features, _, val_latlon, _, val_countries = extract_features(
            feature_extractor,
            val_loader,
            device=device,
            description="Validation features",
        )
        save_checkpoint(
            checkpoint_path,
            args,
            stage="validation_features_ready",
            bank_features=bank_features,
            bank_targets=bank_targets,
            bank_ids=bank_ids,
            bank_countries=bank_countries,
            train_clean_done=True,
            next_aug_index=next_aug_index,
            val_features=val_features,
            val_latlon=val_latlon,
            val_countries=val_countries,
        )

    knn = GeoKNNRegressor(temperature=args.temperature, ref_chunk_size=args.ref_chunk_size)
    knn.fit(bank_features, bank_targets, bank_ids, bank_countries)

    best_metrics: dict[str, float] | None = None
    for k in args.k_values:
        metrics = evaluate_knn(
            knn,
            val_features,
            val_latlon,
            query_countries=val_countries,
            k=k,
            query_chunk_size=args.query_chunk_size,
            country_penalty_multiplier=args.country_penalty_multiplier,
        )
        print(
            f"[VAL] k={k:>2d} | penalized_mean_km={metrics['country_penalized_mean_km']:.2f} | "
            f"mean_km={metrics['mean_km']:.2f} | country_acc={metrics['country_acc']:.3f} | "
            f"acc@25km={metrics['acc_at_25km']:.3f}"
        )
        if best_metrics is None or metrics[SELECTION_METRIC] < best_metrics[SELECTION_METRIC]:
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("validation failed because no k value was evaluated")

    best_k = int(best_metrics["k"])
    print(f"Best k selected on validation with {SELECTION_METRIC}: {best_k}")

    test_metrics: dict[str, float] | None = None
    if not args.skip_test and len(test_samples) > 0:
        test_features, _, test_latlon, _, test_countries = extract_features(
            feature_extractor,
            test_loader,
            device=device,
            description="Test features",
        )
        test_metrics = evaluate_knn(
            knn,
            test_features,
            test_latlon,
            query_countries=test_countries,
            k=best_k,
            query_chunk_size=args.query_chunk_size,
            country_penalty_multiplier=args.country_penalty_multiplier,
        )
        print(
            f"[TEST] k={best_k:>2d} | penalized_mean_km={test_metrics['country_penalized_mean_km']:.2f} | "
            f"mean_km={test_metrics['mean_km']:.2f} | country_acc={test_metrics['country_acc']:.3f} | "
            f"acc@25km={test_metrics['acc_at_25km']:.3f}"
        )

    save_outputs(args.output_path, args.summary_path, knn, best_k, args, best_metrics, test_metrics)
    save_checkpoint(
        checkpoint_path,
        args,
        stage="done",
        bank_features=bank_features,
        bank_targets=bank_targets,
        bank_ids=bank_ids,
        bank_countries=bank_countries,
        train_clean_done=True,
        next_aug_index=next_aug_index,
        val_features=val_features,
        val_latlon=val_latlon,
        val_countries=val_countries,
        best_k=best_k,
        best_metrics=best_metrics,
        test_metrics=test_metrics,
    )

    print(f"Saved KNN index to: {Path(args.output_path).resolve()}")
    print(f"Saved summary to: {Path(args.summary_path).resolve()}")
    print(f"Saved resume checkpoint to: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
