"""
Évaluation du modèle resnet50_classification.pt avec les moyennes GPS réelles
(mean_centers) au lieu des centroïdes k-means.

Prérequis :
    - compute_cells.py doit avoir été relancé pour générer mean_centers dans le pkl
    - Le modèle resnet50_classification.pt doit exister

Usage :
    python eval_mean_centers.py
"""

import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import math

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Modèle (identique à resnet_classification.py)
# ─────────────────────────────────────────────

class GeoResNetClassif(nn.Module):
    def __init__(self, n_cells, hidden_dim=512, dropout_p=0.4):
        super().__init__()
        backbone    = models.resnet50(pretrained=True)
        in_feats    = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_cells),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────
# Dataset (identique à resnet_classification.py)
# ─────────────────────────────────────────────

class GeoDataset(Dataset):
    def __init__(self, csv_file, img_dir, id_to_cell, transform=None,
                 max_samples=None, seed=42):
        data = pd.read_csv(csv_file)
        data = data.dropna(subset=["latitude", "longitude"]).copy()
        data["id"] = data["id"].astype(str)
        data = data[data["id"].isin(id_to_cell)].reset_index(drop=True)

        if max_samples is not None:
            data = data.sample(
                n=min(max_samples, len(data)),
                random_state=seed,
            ).reset_index(drop=True)

        self.data       = data
        self.id_to_cell = id_to_cell
        self.transform  = transform

        self.image_index = {}
        for subdir in os.listdir(img_dir):
            subdir_path = os.path.join(img_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in os.listdir(subdir_path):
                if fname.endswith(".jpg"):
                    self.image_index[fname[:-4]] = os.path.join(subdir_path, fname)

        print(f"Images indexées      : {len(self.image_index)}")
        print(f"Lignes CSV utilisées : {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row    = self.data.iloc[idx]
        img_id = str(row["id"])

        if img_id not in self.image_index:
            raise FileNotFoundError(f"{img_id}.jpg absent du dataset")

        image    = Image.open(self.image_index[img_id]).convert("RGB")
        cell_idx = torch.tensor(self.id_to_cell[img_id], dtype=torch.long)
        latlon   = torch.tensor(
            [float(row["latitude"]), float(row["longitude"])],
            dtype=torch.float32,
        )

        if self.transform:
            image = self.transform(image)

        return image, cell_idx, latlon


# ─────────────────────────────────────────────
# Haversine via centroïdes
# ─────────────────────────────────────────────

def eval_haversine(logits, latlons_true, centers_tensor):
    pred_idx    = logits.argmax(dim=1)
    pred_latlon = centers_tensor[pred_idx]

    lat1 = torch.deg2rad(pred_latlon[:, 0])
    lon1 = torch.deg2rad(pred_latlon[:, 1])
    lat2 = torch.deg2rad(latlons_true[:, 0])
    lon2 = torch.deg2rad(latlons_true[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c    = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return (6371 * c).mean().item()


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CELLS_PATH  = os.path.expanduser("~/datasets/OSV5M/cells_kmeans.pkl")
MODEL_PATH  = "models/resnet/resnet50_classification.pt"
MAX_SAMPLES = None
SEED        = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    vram       = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    BATCH_SIZE = 32 if vram >= 8 else 16
else:
    BATCH_SIZE = 16

print(f"Device: {device} | Batch size: {BATCH_SIZE}")

# ─────────────────────────────────────────────
# Chargement des cellules
# ─────────────────────────────────────────────

print(f"\nChargement des cellules : {CELLS_PATH}")
with open(CELLS_PATH, "rb") as f:
    cells = pickle.load(f)

if "mean_centers" not in cells:
    raise KeyError(
        "mean_centers absent du pkl — relance compute_cells.py d'abord."
    )

n_cells    = cells["n_cells"]
id_to_cell = cells["id_to_cell"]

centers_kmeans = torch.tensor(cells["centers"],      dtype=torch.float32).to(device)
centers_mean   = torch.tensor(cells["mean_centers"], dtype=torch.float32).to(device)

print(f"Cellules chargées : {n_cells}")
print(f"centers      (k-means) : OK")
print(f"mean_centers (GPS réel) : OK")

# ─────────────────────────────────────────────
# Dataset — même split 90/10 que l'entraînement
# ─────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = GeoDataset(
    csv_file    = os.path.expanduser("~/datasets/OSV5M/test_filtered.csv"),
    img_dir     = os.path.expanduser("~/datasets/OSV5M"),
    id_to_cell  = id_to_cell,
    transform   = transform,
    max_samples = MAX_SAMPLES,
    seed        = SEED,
)

train_size = int(0.9 * len(dataset))
test_size  = len(dataset) - train_size
_, test_ds = random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED),
)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

print(f"Jeu de test : {len(test_ds)} images")

# ─────────────────────────────────────────────
# Chargement du modèle
# ─────────────────────────────────────────────

print(f"\nChargement du modèle : {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)

model = GeoResNetClassif(
    n_cells    = n_cells,
    hidden_dim = checkpoint["config"]["hidden_dim"],
    dropout_p  = checkpoint["config"]["dropout_p"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Modèle chargé.")

# ─────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────

km_kmeans = 0.0
km_mean   = 0.0
correct   = 0
n_total   = 0
n_batches = 0

with torch.no_grad():
    for imgs, cell_idx, latlons in test_loader:
        imgs     = imgs.to(device)
        cell_idx = cell_idx.to(device)
        latlons  = latlons.to(device)

        logits = model(imgs)

        km_kmeans += eval_haversine(logits, latlons, centers_kmeans)
        km_mean   += eval_haversine(logits, latlons, centers_mean)
        correct   += (logits.argmax(1) == cell_idx).sum().item()
        n_total   += len(imgs)
        n_batches += 1

km_kmeans /= n_batches
km_mean   /= n_batches
acc        = 100 * correct / n_total

# ─────────────────────────────────────────────
# Résultats
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("RÉSULTATS")
print("=" * 50)
print(f"Accuracy cellule          : {acc:.1f}%")
print(f"Haversine (centroïde k-means) : {km_kmeans:.1f} km")
print(f"Haversine (moyenne GPS réelle) : {km_mean:.1f} km")
gain = km_kmeans - km_mean
print(f"Gain                      : {gain:.1f} km ({100*gain/km_kmeans:.1f}%)")
