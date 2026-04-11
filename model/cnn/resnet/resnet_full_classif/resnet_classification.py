"""
Géolocalisation : classification pure en cellules géographiques k-means.

Pipeline :
    1. Les coordonnées GPS sont converties en indices de cellules k-means
       pré-calculées (compute_cells.py doit avoir été lancé avant).
    2. ResNet50 classifie l'image en l'une des N_CELLS cellules.
    3. Loss : CrossEntropy pondérée par la distance géographique entre cellules
       (soft labels, les cellules voisines reçoivent une probabilité cible non nulle).
    4. La prédiction GPS finale est le centroïde de la cellule prédite.
    5. Évaluation : distance Haversine entre centroïde prédit et vraies coordonnées.

Usage :
    python resnet_classification.py
"""

import os
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Utilitaires géographiques
# ─────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Distance Haversine en km entre deux points (degrés)."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_matrix(centers):
    """
    Calcule la matrice (N_CELLS x N_CELLS) des distances Haversine entre centroïdes.
    Retourne un tenseur float32.
    """
    n    = len(centers)
    lats = torch.tensor(centers[:, 0], dtype=torch.float32)
    lons = torch.tensor(centers[:, 1], dtype=torch.float32)

    lat1 = lats.unsqueeze(1).expand(n, n)
    lon1 = lons.unsqueeze(1).expand(n, n)
    lat2 = lats.unsqueeze(0).expand(n, n)
    lon2 = lons.unsqueeze(0).expand(n, n)

    lat1_r = torch.deg2rad(lat1)
    lon1_r = torch.deg2rad(lon1)
    lat2_r = torch.deg2rad(lat2)
    lon2_r = torch.deg2rad(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = torch.sin(dlat/2)**2 + torch.cos(lat1_r) * torch.cos(lat2_r) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return 6371 * c   # (N_CELLS, N_CELLS) en km


# ─────────────────────────────────────────────
# Soft label loss (CrossEntropy géographique)
# ─────────────────────────────────────────────

class GeoSoftCrossEntropy(nn.Module):
    """
    CrossEntropy avec soft labels géographiques.

    Pour chaque image, la cellule vraie reçoit le poids principal.
    Les cellules voisines (distance < sigma km) reçoivent un poids décroissant
    selon une gaussienne. Cela pénalise moins les erreurs géographiquement proches.

    Paramètres
    ----------
    dist_matrix : tenseur (N_CELLS, N_CELLS) des distances en km
    sigma       : écart-type de la gaussienne en km (défaut 750 km)
    """

    def __init__(self, dist_matrix, sigma=750):
        super().__init__()
        # Poids gaussiens normalisés : (N_CELLS, N_CELLS)
        weights = torch.exp(-dist_matrix**2 / (2 * sigma**2))
        weights = weights / weights.sum(dim=1, keepdim=True)
        self.register_buffer("weights", weights)

    def forward(self, logits, targets):
        # targets : (B,) indices de cellules
        soft_targets = self.weights[targets]                   # (B, N_CELLS)
        log_probs    = F.log_softmax(logits, dim=1)            # (B, N_CELLS)
        loss         = -(soft_targets * log_probs).sum(dim=1)  # (B,)
        return loss.mean()


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class GeoDataset(Dataset):
    """
    Charge les images OSV5M et retourne :
        image, cell_idx (int), (lat, lon) vraies pour l'évaluation Haversine
    """

    def __init__(self, csv_file, img_dir, id_to_cell, transform=None,
                 max_samples=None, seed=42):
        data = pd.read_csv(csv_file)
        data = data.dropna(subset=["latitude", "longitude"]).copy()
        data["id"] = data["id"].astype(str)

        # Filtre les images sans cellule connue
        data = data[data["id"].isin(id_to_cell)].reset_index(drop=True)

        if max_samples is not None:
            data = data.sample(
                n=min(max_samples, len(data)),
                random_state=seed,
            ).reset_index(drop=True)

        self.data       = data
        self.id_to_cell = id_to_cell
        self.transform  = transform

        # Index fichiers image
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

        image     = Image.open(self.image_index[img_id]).convert("RGB")
        cell_idx  = torch.tensor(self.id_to_cell[img_id], dtype=torch.long)
        latlon    = torch.tensor([float(row["latitude"]), float(row["longitude"])],
                                 dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, cell_idx, latlon


# ─────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────

class GeoResNetClassif(nn.Module):
    """
    ResNet50 → classification en N_CELLS cellules géographiques.
    La prédiction GPS est le centroïde de la cellule prédite.
    """

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
        feats  = self.backbone(x)
        logits = self.head(feats)
        return logits


# ─────────────────────────────────────────────
# Évaluation Haversine via centroïdes
# ─────────────────────────────────────────────

def eval_haversine(logits, latlons_true, centers_tensor):
    """
    Convertit les logits en centroïdes prédits et calcule la distance Haversine
    moyenne en km par rapport aux vraies coordonnées.

    Paramètres
    ----------
    logits        : (B, N_CELLS)
    latlons_true  : (B, 2) en degrés
    centers_tensor: (N_CELLS, 2) en degrés
    """
    pred_idx    = logits.argmax(dim=1)                        # (B,)
    pred_latlon = centers_tensor[pred_idx]                    # (B, 2)

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

CONFIG = {
    "hidden_dim"  : 512,
    "dropout_p"   : 0.4,
    "lr"          : 1.3e-4,
    "weight_decay": 1.9e-4,
    "sigma_km"    : 750,      # écart-type de la gaussienne soft label
    "unfreeze_all": True,
}

FINAL_EPOCHS = 30
MAX_SAMPLES  = 500_000        # mettre None pour tout le dataset
CELLS_PATH   = os.path.expanduser("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/resnet/resnet_full_classif/cells_kmeans.pkl")

# ─────────────────────────────────────────────
# Setup device
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    vram       = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    BATCH_SIZE = 32 if vram >= 8 else 16
else:
    BATCH_SIZE = 16

print(f"Device: {device} | Batch size: {BATCH_SIZE}")

# ─────────────────────────────────────────────
# Chargement des cellules k-means
# ─────────────────────────────────────────────

print(f"Chargement des cellules : {CELLS_PATH}")
with open(CELLS_PATH, "rb") as f:
    cells = pickle.load(f)

n_cells        = cells["n_cells"]
centers        = cells["centers"]          # (N_CELLS, 2) numpy float32
id_to_cell     = cells["id_to_cell"]
centers_tensor = torch.tensor(centers, dtype=torch.float32).to(device)

print(f"Cellules chargées : {n_cells}")

# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# Dataset & DataLoaders
# ─────────────────────────────────────────────

dataset = GeoDataset(
    csv_file    = os.path.expanduser("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/rest_filtered_v2.csv"),
    img_dir     = os.path.expanduser("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/images/rest_images"),
    id_to_cell  = id_to_cell,
    transform   = transform,
    max_samples = MAX_SAMPLES,
)


# ─────────────────────────────────────────────
# Undersampling US
# ─────────────────────────────────────────────

US_CAP = 30_000

# Ajouter la colonne country si absente
if "country" not in dataset.data.columns:
    csv_country = pd.read_csv(
        "/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/rest_filtered_v2.csv",
        usecols=["id", "country"]
    )
    csv_country["id"] = csv_country["id"].astype(str)
    dataset.data = dataset.data.merge(csv_country, on="id", how="left")

us_mask     = dataset.data["country"] == "US"
non_us_data = dataset.data[~us_mask].copy()
us_data     = dataset.data[us_mask].copy()

if len(us_data) > US_CAP:
    us_data = us_data.sample(n=US_CAP, random_state=42).reset_index(drop=True)
    print(f"Undersampling US : {US_CAP} / {us_mask.sum()} gardés")

dataset.data = pd.concat([non_us_data, us_data], ignore_index=True)
print(f"Dataset après undersampling : {len(dataset.data)} samples")

# ─────────────────────────────────────────────
# Split train / test
# ─────────────────────────────────────────────

train_size = int(0.9 * len(dataset))
test_size  = len(dataset) - train_size
train_ds, test_ds = random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42),
)


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# ─────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────

cfg   = CONFIG
model = GeoResNetClassif(
    n_cells    = n_cells,
    hidden_dim = cfg["hidden_dim"],
    dropout_p  = cfg["dropout_p"],
).to(device)

if cfg["unfreeze_all"]:
    for param in model.parameters():
        param.requires_grad = True

# ─────────────────────────────────────────────
# Loss, optimiseur, scheduler
# ─────────────────────────────────────────────

# Matrice de distances entre centroïdes (calculée une fois)
print("Calcul de la matrice de distances entre centroïdes...")
dist_matrix = haversine_matrix(centers).to(device)
criterion   = GeoSoftCrossEntropy(dist_matrix, sigma=cfg["sigma_km"])
print("Matrice prête.")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr           = cfg["lr"],
    weight_decay = cfg["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

# ─────────────────────────────────────────────
# Boucle d'entraînement
# ─────────────────────────────────────────────

best_test_km  = float("inf")
best_test_acc = 0.0

print(f"\nEntraînement, {FINAL_EPOCHS} epochs | "
      f"{MAX_SAMPLES if MAX_SAMPLES else 'tous les'} samples | "
      f"{n_cells} cellules\n")

for epoch in range(FINAL_EPOCHS):

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    train_loss_total = 0.0
    train_correct    = 0
    train_n          = 0

    for imgs, cell_idx, latlons in train_loader:
        imgs     = imgs.to(device)
        cell_idx = cell_idx.to(device)
        latlons  = latlons.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, cell_idx)

        if torch.isnan(loss):
            print(f"  [WARN] NaN à l'epoch {epoch+1}, batch ignoré")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_loss_total += loss.item()
        train_correct    += (logits.argmax(1) == cell_idx).sum().item()
        train_n          += len(imgs)

    # ── Eval ───────────────────────────────────────────────────────────────────
    model.eval()
    test_loss_total = 0.0
    test_km_total   = 0.0
    test_correct    = 0
    test_n          = 0
    n_batches       = 0

    with torch.no_grad():
        for imgs, cell_idx, latlons in test_loader:
            imgs     = imgs.to(device)
            cell_idx = cell_idx.to(device)
            latlons  = latlons.to(device)

            logits = model(imgs)
            test_loss_total += criterion(logits, cell_idx).item()
            test_km_total   += eval_haversine(logits, latlons, centers_tensor)
            test_correct    += (logits.argmax(1) == cell_idx).sum().item()
            test_n          += len(imgs)
            n_batches       += 1

    train_loss = train_loss_total / len(train_loader)
    test_loss  = test_loss_total  / len(test_loader)
    test_km    = test_km_total    / n_batches
    train_acc  = 100 * train_correct / train_n
    test_acc   = 100 * test_correct  / test_n

    if test_km < best_test_km:
        best_test_km  = test_km
        best_test_acc = test_acc

    scheduler.step()

    print(
        f"Epoch {epoch+1:2d}/{FINAL_EPOCHS} | "
        f"Train loss: {train_loss:.3f}  acc: {train_acc:.1f}% | "
        f"Test loss: {test_loss:.3f}  acc: {test_acc:.1f}%  "
        f"Haversine: {test_km:.1f} km"
    )

print(f"\nMeilleur test, {best_test_km:.1f} km | acc cellule {best_test_acc:.1f}%")

# ─────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────

#MODEL_DIR = Path("models/resnet")
#MODEL_DIR.mkdir(parents=True, exist_ok=True)
save_path = "resnet50_classification_full_dataset_undersampling.pt"

torch.save({
    "model_state_dict"    : model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epochs"              : FINAL_EPOCHS,
    "best_test_km"        : best_test_km,
    "best_test_acc"       : best_test_acc,
    "n_cells"             : n_cells,
    "centers"             : centers,
    "config"              : cfg,
}, save_path)

print(f"Modèle sauvegardé : {save_path}")
