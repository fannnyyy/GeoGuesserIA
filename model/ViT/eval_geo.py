# eval_geo.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


DATASET_DIR = Path("dataset/osv5m_test")
CSV_PATH = DATASET_DIR / "test_filtered.csv"
CKPT_PATH = Path("checkpoints/vit_geo_best.pt")  # ou checkpoints/vit_geo.pt
BATCH_SIZE = 64
NUM_WORKERS = 8


def latlon_to_unitvec(lat_deg, lon_deg) -> torch.Tensor:
    lat = torch.deg2rad(torch.tensor(lat_deg, dtype=torch.float32))
    lon = torch.deg2rad(torch.tensor(lon_deg, dtype=torch.float32))
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def normalize_vec(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


def unitvec_to_latlon(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v = normalize_vec(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    lat = torch.asin(z)
    lon = torch.atan2(y, x)
    return torch.rad2deg(lat), torch.rad2deg(lon)


def haversine_km(lat1, lon1, lat2, lon2) -> torch.Tensor:
    R = 6371.0
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.clamp(a.sqrt(), 0.0, 1.0))
    return R * c


def cosine_loss(pred, target):
    pred = normalize_vec(pred)
    target = normalize_vec(target)
    return 1.0 - (pred * target).sum(dim=-1).mean()


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
        y = self.id_to_vec[img_id]
        return x, y


class ViTGeo(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vit_b_16(weights=None)
        self.net.heads.head = nn.Linear(self.net.heads.head.in_features, 3)

    def forward(self, x):
        return self.net(x)


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)
    df["id"] = df["id"].astype(str)

    vecs = latlon_to_unitvec(df["latitude"].values, df["longitude"].values)
    id_to_vec = dict(zip(df["id"].tolist(), vecs))

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    base_dataset = datasets.ImageFolder(root=str(DATASET_DIR), transform=tfm)
    geo_dataset = GeoRegressionWrapper(base_dataset, id_to_vec)

    # même split que dans train_vit_geo.py
    n = len(geo_dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    _, _, test_ds = random_split(
        geo_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = ViTGeo().to(device)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    losses = []
    dists = []

    with torch.no_grad():
        for x, y_true_vec in tqdm(test_loader, desc="Eval"):
            x = x.to(device)
            y_true_vec = y_true_vec.to(device)

            y_pred_vec = normalize_vec(model(x))

            losses.append(cosine_loss(y_pred_vec, y_true_vec).item())

            lat_t, lon_t = unitvec_to_latlon(y_true_vec)
            lat_p, lon_p = unitvec_to_latlon(y_pred_vec)

            dist = haversine_km(lat_t, lon_t, lat_p, lon_p)
            dists.append(dist.mean().item())

    print("Test cosine loss (avg):", sum(losses) / len(losses))
    print("Mean Haversine distance (km):", sum(dists) / len(dists))


if __name__ == "__main__":
    main()