# train_vit_geo.py
# Entraînement ViT pour régression géographique (vecteur unité 3D) + reprise via checkpoint Slurm-friendly

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


# =========================
# CONFIG
# =========================
DATASET_DIR = Path("dataset/osv5m_test")
CSV_PATH = DATASET_DIR / "test_filtered.csv"

BATCH_SIZE = 32
NUM_WORKERS = 8          # mets 4 si ça sature / si le cluster limite
LR = 1e-5
EPOCHS = 150
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

CKPT_PATH = Path("checkpoints/vit_geo.pt")

# Mapping check (à faire 1 fois, puis laisser False pour les runs longs)
DO_MAPPING_CHECK = False
MAPPING_CHECK_MAX_SHOW = 3


# =========================
# UTILS
# =========================
def extract_id_from_filename(path: Path) -> str | None:
    m = re.findall(r"\d+", path.stem)
    if not m:
        return None
    return max(m, key=len)

def cosine_loss(pred, target):
    pred = normalize_vec(pred)
    target = normalize_vec(target)
    # 1 - cos(theta) : 0 si parfait
    return 1.0 - (pred * target).sum(dim=-1).mean()

def latlon_to_unitvec(lat_deg, lon_deg) -> torch.Tensor:
    lat = torch.deg2rad(torch.tensor(lat_deg, dtype=torch.float32))
    lon = torch.deg2rad(torch.tensor(lon_deg, dtype=torch.float32))
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)  # (..., 3)


def normalize_vec(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> float:
    model.eval()
    tot = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = normalize_vec(model(x))
        tot += loss_fn(pred, y).item()
    return tot / max(1, len(loader))


# =========================
# DATASET WRAPPER
# =========================
class GeoRegressionWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset: datasets.ImageFolder, id_to_vec: dict[str, torch.Tensor]):
        self.base = base_dataset
        self.id_to_vec = id_to_vec

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        path, _ = self.base.samples[idx]
        img_id = Path(path).stem
        y = self.id_to_vec[img_id]  # tensor (3,)
        return x, y


# =========================
# MODEL
# =========================
class ViTGeo(nn.Module):
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        self.net = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in self.net.parameters():
                p.requires_grad = False
        self.net.heads.head = nn.Linear(self.net.heads.head.in_features, 3)

    def forward(self, x):
        return self.net(x)


def set_trainable_vit(model: ViTGeo, unfreeze_last_n_blocks: int = 12):
    # Tout geler
    for p in model.net.parameters():
        p.requires_grad = False

    # Tête toujours trainable
    for p in model.net.heads.head.parameters():
        p.requires_grad = True

    # Unfreeze derniers blocks
    layers = model.net.encoder.layers
    n = len(layers)
    k = max(0, min(unfreeze_last_n_blocks, n))
    for i in range(n - k, n):
        for p in layers[i].parameters():
            p.requires_grad = True

    # LayerNorm final + conv_proj (souvent utile)
    for p in model.net.encoder.ln.parameters():
        p.requires_grad = True
    for p in model.net.conv_proj.parameters():
        p.requires_grad = True


def count_trainable_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# =========================
# TRAIN (RESUME)
# =========================
def train_resume(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    ckpt_path: Path,
    device: torch.device,
    grad_clip: float | None = 1.0,
    resume: bool = True,
    unfreeze_schedule: dict[int, int] | None = None,
):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    loss_fn = cosine_loss

    start_epoch = 0
    best_val = float("inf")

    optimizer = None
    scheduler = None

    def rebuild_optim(epoch_idx: int):
        nonlocal optimizer, scheduler
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        total_steps = (epochs - epoch_idx) * max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
        )

    # 1) charger le checkpoint d'abord
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"[RESUME] {ckpt_path} from epoch {start_epoch} | best_val={best_val:.6f}")

    # 2) appliquer ensuite le bon unfreeze selon start_epoch
    if isinstance(model, ViTGeo) and unfreeze_schedule:
        applicable = [e for e in unfreeze_schedule.keys() if e <= start_epoch]
        if applicable:
            e0 = max(applicable)
            set_trainable_vit(model, unfreeze_last_n_blocks=unfreeze_schedule[e0])
        else:
            set_trainable_vit(model, unfreeze_last_n_blocks=min(unfreeze_schedule.values()))

    # 3) construire optimizer après avoir fixé requires_grad
    rebuild_optim(start_epoch)




    # Sanity GPU
    print("device:", device)
    if device.type != "cuda":
        print("⚠️ CUDA non détecté. Assure-toi de lancer via Slurm sur une partition GPU avec --gres=gpu:1.")

    epoch_pbar = tqdm(range(start_epoch, epochs), desc="Epochs", leave=True)

    for epoch in epoch_pbar:
        # Progressive unfreeze
        if isinstance(model, ViTGeo) and unfreeze_schedule and epoch in unfreeze_schedule:
            set_trainable_vit(model, unfreeze_last_n_blocks=unfreeze_schedule[epoch])
            rebuild_optim(epoch)
            tr, tot = count_trainable_params(model)
            print(
                f"[UNFREEZE] epoch {epoch}: last_n_blocks={unfreeze_schedule[epoch]} | "
                f"trainable={tr/1e6:.2f}M / {tot/1e6:.2f}M"
            )

        assert optimizer is not None and scheduler is not None

        model.train()
        running = 0.0

        batch_pbar = tqdm(train_loader, desc=f"Train (epoch {epoch+1})", leave=False)
        for x, y in batch_pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = normalize_vec(model(x))
            
            loss = cosine_loss(pred, y)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            running += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = running / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, loss_fn, device)

        epoch_pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        # Save last
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "best_val": best_val,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpt_path,
        )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_val": best_val,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                ckpt_path.with_name(ckpt_path.stem + "_best.pt"),
            )

    print(f"Done. best_val={best_val:.6f}")
    return model


# =========================
# MAIN
# =========================
def main():
    # --- Load CSV once ---
    df = pd.read_csv(CSV_PATH)
    df["id"] = df["id"].astype(str)

    print("CSV rows:", len(df))
    print("CSV columns:", df.columns.tolist())

    # --- Optional mapping check (expensive) ---
    if DO_MAPPING_CHECK:
        image_dirs = [DATASET_DIR / f"{i:02d}" for i in range(5)]
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        images = []
        for d in image_dirs:
            if d.exists():
                images += [p for p in d.rglob("*") if p.suffix.lower() in exts]

        print("Nb images trouvées:", len(images))
        print("Exemple image:", images[0] if images else None)

        missing_id_in_filename = []
        no_row_for_image = []
        multiple_rows_for_image = []
        mapped = []
        id_to_rows = df.groupby("id").size()

        for p in images:
            img_id = extract_id_from_filename(p)
            if img_id is None:
                missing_id_in_filename.append(str(p))
                continue

            count = int(id_to_rows.get(img_id, 0))
            if count == 0:
                no_row_for_image.append((str(p), img_id))
            elif count > 1:
                multiple_rows_for_image.append((str(p), img_id, count))
            else:
                mapped.append((str(p), img_id))

        print("\n--- Résumé mapping ---")
        print("Images sans ID détectable dans le nom:", len(missing_id_in_filename))
        print("Images avec ID mais aucune ligne CSV:", len(no_row_for_image))
        print("Images avec plusieurs lignes CSV pour le même ID:", len(multiple_rows_for_image))
        print("Images correctement mappées (1 ligne):", len(mapped))

        if missing_id_in_filename[:MAPPING_CHECK_MAX_SHOW]:
            print("\nExemples: pas d'ID dans filename")
            for x in missing_id_in_filename[:MAPPING_CHECK_MAX_SHOW]:
                print(" -", x)

        if no_row_for_image[:MAPPING_CHECK_MAX_SHOW]:
            print("\nExemples: ID pas trouvé dans CSV")
            for path, img_id in no_row_for_image[:MAPPING_CHECK_MAX_SHOW]:
                print(" -", img_id, "->", path)

        if multiple_rows_for_image[:MAPPING_CHECK_MAX_SHOW]:
            print("\nExemples: ID dupliqué dans CSV")
            for path, img_id, c in multiple_rows_for_image[:MAPPING_CHECK_MAX_SHOW]:
                print(" -", img_id, f"(x{c}) ->", path)

        if mapped:
            sample_path, sample_id = mapped[0]
            row = df[df["id"] == sample_id].iloc[0]
            print("\n--- Exemple mapping OK ---")
            print("Image:", sample_path)
            print("ID:", sample_id)
            print("lat/lon:", row["latitude"], row["longitude"])
            print("country:", row.get("country", None), "| city:", row.get("city", None))

    # --- Build labels dict (fast, vectorized) ---
    vecs = latlon_to_unitvec(df["latitude"].values, df["longitude"].values)  # (N,3)
    id_to_vec = dict(zip(df["id"].tolist(), vecs))

    # --- Image transforms ---
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    # ImageFolder: 00..04 treated as classes, ignored later
    base_dataset = datasets.ImageFolder(root=str(DATASET_DIR), transform=tfm)
    geo_dataset = GeoRegressionWrapper(base_dataset, id_to_vec)

    print("Dataset size:", len(geo_dataset))
    x0, y0 = geo_dataset[0]
    print("Example:", x0.shape, y0)

    # --- Split ---
    n = len(geo_dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        geo_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print("Splits:", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Model (ViT) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = ViTGeo(freeze_backbone=False)
  

    tr, tot = count_trainable_params(vit)
    print(f"ViT params: {tr/1e6:.2f}M / {tot/1e6:.2f}M")

    unfreeze_schedule = {
        0: 12,
    }

    vit = train_resume(
        vit,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        ckpt_path=CKPT_PATH,
        device=device,
        grad_clip=GRAD_CLIP,
        resume=True,
        unfreeze_schedule=unfreeze_schedule,
    )

    # Save final model weights (optional)
    torch.save(vit.state_dict(), CKPT_PATH.with_name("vit_geo_final_state_dict.pt"))
    print("Saved:", CKPT_PATH.with_name("vit_geo_final_state_dict.pt"))

    # Note: evaluation Haversine can be done in a separate script/cell to keep training fast.


if __name__ == "__main__":
    import os
    print("CWD:", os.getcwd())
    print("DATASET_DIR exists:", DATASET_DIR.exists(), DATASET_DIR.resolve())
    print("CSV_PATH exists:", CSV_PATH.exists(), CSV_PATH.resolve())
    main()