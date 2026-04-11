"""
Géolocalisation : classification pays + régression GPS (pipeline joint)

Architecture :
    - Backbone ResNet50 partagé
    - Tête 1 : classification → N pays (CrossEntropy)
    - Tête 2 : régression GPS sin/cos, conditionnée par l'embedding pays
    - Loss totale : λ_cls * CE + λ_reg * Haversine

Curriculum lambda :
    - Epochs 1..PHASE1_EPOCHS  : λ_cls élevé → le modèle apprend d'abord les pays
    - Epochs PHASE1_EPOCHS+1.. : λ_cls faible → la régression GPS prend le dessus

Usage :
    python resnet_classif_regress.py
"""

import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class GeoDataset(Dataset):
    """
    Charge les images OSV5M et retourne :
        image, coords (sin/cos lat/lon), country_idx (int)

    Paramètres
    ----------
    csv_file     : chemin vers le CSV (doit contenir id, latitude, longitude, country)
    img_dir      : dossier racine contenant les sous-dossiers d'images
    transform    : pipeline torchvision
    max_samples  : limite optionnelle du nombre de lignes
    seed         : graine aléatoire pour l'échantillonnage
    """

    def __init__(self, csv_file, img_dir, transform=None, max_samples=None, seed=42):
        data = pd.read_csv(csv_file)

        # ── Colonne pays OSV5M ────────────────────────────────────────────────
        # OSV5M fournit la colonne "country" avec des codes ISO alpha-2
        # (ex. 'FR' = France, 'MG' = Madagascar). On filtre les NaN avant tout.
        data = data.dropna(subset=["country"]).copy()

        if max_samples is not None:
            data = data.sample(
                n=min(max_samples, len(data)),
                random_state=seed,
            ).reset_index(drop=True)

        # Encodage code ISO → indice continu 0..N-1
        countries = sorted(data["country"].unique().tolist())
        self.country_to_idx = {c: i for i, c in enumerate(countries)}
        self.num_countries  = len(countries)
        print(f"Pays distincts : {self.num_countries}")

        self.data      = data
        self.transform = transform

        # Index fichiers image
        self.image_index = {}
        for subdir in os.listdir(img_dir):
            subdir_path = os.path.join(img_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in os.listdir(subdir_path):
                if fname.endswith(".jpg"):
                    self.image_index[fname[:-4]] = os.path.join(subdir_path, fname)

        print(f"Images indexées  : {len(self.image_index)}")
        print(f"Lignes CSV utilisées : {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row    = self.data.iloc[idx]
        img_id = str(row["id"])

        if img_id not in self.image_index:
            raise FileNotFoundError(f"{img_id}.jpg absent du dataset")

        image = Image.open(self.image_index[img_id]).convert("RGB")

        # Coordonnées sin/cos
        lat_r  = math.radians(float(row["latitude"]))
        lon_r  = math.radians(float(row["longitude"]))
        coords = torch.tensor(
            [math.sin(lat_r), math.cos(lat_r), math.sin(lon_r), math.cos(lon_r)],
            dtype=torch.float32,
        )

        country_idx = torch.tensor(self.country_to_idx[row["country"]], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, coords, country_idx


# ─────────────────────────────────────────────
# Loss Haversine
# ─────────────────────────────────────────────

def sincos_to_rad(sin, cos):
    return torch.atan2(sin, cos)


class HaversineLoss(nn.Module):
    def __init__(self, radius=6371):
        super().__init__()
        self.radius = radius

    def forward(self, preds, targets):
        lat1 = sincos_to_rad(preds[:, 0],   preds[:, 1])
        lon1 = sincos_to_rad(preds[:, 2],   preds[:, 3])
        lat2 = sincos_to_rad(targets[:, 0], targets[:, 1])
        lon2 = sincos_to_rad(targets[:, 2], targets[:, 3])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        )
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return (self.radius * c).mean()


# ─────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────

class GeoResNetClassifRegress(nn.Module):
    """
    Backbone ResNet50 partagé avec deux têtes :

    1. Tête classification (cls_head)
       Linear(2048 → hidden) → ReLU → Dropout → Linear(→ num_countries)
       → logits pays (CrossEntropy)
       → softmax → embedding pays (dim = num_countries) injecté dans la régression

    2. Tête régression (reg_head)
       Entrée = concat(features_backbone, embedding_pays)
                = 2048 + num_countries
       Linear → ReLU → Dropout → Linear(→ 4)
       → normalisation par paires sin/cos

    Paramètres
    ----------
    num_countries  : nombre de classes pays
    hidden_dim     : dimension cachée des deux têtes
    dropout_p      : taux de dropout
    embed_detach   : si True, l'embedding pays est détaché du graphe de grad
                     avant concaténation (la classification ne reçoit pas de
                     signal du gradient de la régression)
    """

    def __init__(
        self,
        num_countries: int,
        hidden_dim: int = 512,
        dropout_p: float = 0.4,
        embed_detach: bool = False,
    ):
        super().__init__()
        self.embed_detach = embed_detach

        # ── Backbone ──────────────────────────────────────────────────────────
        backbone    = models.resnet50(pretrained=True)
        in_feats    = backbone.fc.in_features          # 2048
        backbone.fc = nn.Identity()                    # supprime la tête originale
        self.backbone = backbone

        # ── Tête 1 : classification ────────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_countries),
        )

        # ── Tête 2 : régression conditionnée par le pays ───────────────────────
        # Entrée = features backbone (2048) + proba pays (num_countries)
        self.reg_head = nn.Sequential(
            nn.Linear(in_feats + num_countries, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x):
        # 1. Features partagées
        feats = self.backbone(x)                          # (B, 2048)

        # 2. Logits et probabilités pays
        cls_logits = self.cls_head(feats)                 # (B, num_countries)
        cls_probs  = torch.softmax(cls_logits, dim=1)     # (B, num_countries)

        # 3. Embedding pays (optionnellement détaché)
        embed = cls_probs.detach() if self.embed_detach else cls_probs

        # 4. Régression conditionnée
        reg_input  = torch.cat([feats, embed], dim=1)     # (B, 2048 + num_countries)
        raw_coords = self.reg_head(reg_input)              # (B, 4)

        # 5. Normalisation sin/cos sur le cercle unité
        lat    = F.normalize(raw_coords[:, 0:2], dim=1)
        lon    = F.normalize(raw_coords[:, 2:4], dim=1)
        coords = torch.cat([lat, lon], dim=1)              # (B, 4)

        return coords, cls_logits


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CONFIG = {
    "hidden_dim":      512,
    "dropout_p":       0.4,
    "lr":              1.3e-4,
    "weight_decay":    1.9e-4,
    "embed_detach":    False,
    # Curriculum lambda :
    "lambda_cls_high": 10,    # epochs 1..PHASE1_EPOCHS
    "lambda_cls_low":  1,     # epochs PHASE1_EPOCHS+1..fin
    "lambda_reg":      1.0,
    "unfreeze_all":    True,
}

FINAL_EPOCHS  = 30
PHASE1_EPOCHS = 10    # nombre d'epochs avec lambda_cls élevé
MAX_SAMPLES   = None

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
    csv_file    = os.path.expanduser("~/datasets/OSV5M/test_filtered.csv"),
    img_dir     = os.path.expanduser("~/datasets/OSV5M"),
    transform   = transform,
    max_samples = MAX_SAMPLES,
)

train_size = int(0.9 * len(dataset))
test_size  = len(dataset) - train_size
train_ds, test_ds = random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ─────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────

cfg   = CONFIG
model = GeoResNetClassifRegress(
    num_countries = dataset.num_countries,
    hidden_dim    = cfg["hidden_dim"],
    dropout_p     = cfg["dropout_p"],
    embed_detach  = cfg["embed_detach"],
).to(device)

# Gel / dégel
if cfg["unfreeze_all"]:
    for param in model.parameters():
        param.requires_grad = True
else:
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.cls_head.parameters():
        param.requires_grad = True
    for param in model.reg_head.parameters():
        param.requires_grad = True

# ─────────────────────────────────────────────
# Optimiseur & scheduler
# ─────────────────────────────────────────────

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr           = cfg["lr"],
    weight_decay = cfg["weight_decay"],
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

criterion_reg = HaversineLoss()
criterion_cls = nn.CrossEntropyLoss()

lambda_reg      = cfg["lambda_reg"]
lambda_cls_high = cfg["lambda_cls_high"]
lambda_cls_low  = cfg["lambda_cls_low"]

# ─────────────────────────────────────────────
# Boucle d'entraînement
# ─────────────────────────────────────────────

best_test_km  = float("inf")
best_test_acc = 0.0

print(f"\nEntraînement, {FINAL_EPOCHS} epochs | {MAX_SAMPLES} samples")
print(f"Pays : {dataset.num_countries} | "
      f"λ_cls {lambda_cls_high} (epochs 1-{PHASE1_EPOCHS}) "
      f"→ {lambda_cls_low} (epochs {PHASE1_EPOCHS+1}-{FINAL_EPOCHS})\n")

for epoch in range(FINAL_EPOCHS):

    # ── Curriculum : choix du lambda_cls selon la phase ───────────────────────
    lambda_cls = lambda_cls_high if epoch < PHASE1_EPOCHS else lambda_cls_low

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    train_reg_total = 0.0
    train_cls_total = 0.0
    train_correct   = 0
    train_n         = 0

    for imgs, coords, country_idx in train_loader:
        imgs        = imgs.to(device)
        coords      = coords.to(device)
        country_idx = country_idx.to(device)

        optimizer.zero_grad()
        pred_coords, cls_logits = model(imgs)

        loss_reg = criterion_reg(pred_coords, coords)
        loss_cls = criterion_cls(cls_logits, country_idx)
        loss     = lambda_reg * loss_reg + lambda_cls * loss_cls

        # Détection NaN avant backward pour éviter la corruption des poids
        if torch.isnan(loss):
            print(f"  [WARN] NaN détecté à l'epoch {epoch+1}, batch ignoré")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_reg_total += loss_reg.item()
        train_cls_total += loss_cls.item()
        train_correct   += (cls_logits.argmax(1) == country_idx).sum().item()
        train_n         += len(imgs)

    # ── Eval ───────────────────────────────────────────────────────────────────
    model.eval()
    test_reg_total = 0.0
    test_correct   = 0
    test_n         = 0

    with torch.no_grad():
        for imgs, coords, country_idx in test_loader:
            imgs        = imgs.to(device)
            coords      = coords.to(device)
            country_idx = country_idx.to(device)

            pred_coords, cls_logits = model(imgs)
            test_reg_total += criterion_reg(pred_coords, coords).item()
            test_correct   += (cls_logits.argmax(1) == country_idx).sum().item()
            test_n         += len(imgs)

    train_km  = train_reg_total / len(train_loader)
    test_km   = test_reg_total  / len(test_loader)
    train_acc = 100 * train_correct / train_n
    test_acc  = 100 * test_correct  / test_n

    if test_km < best_test_km:
        best_test_km  = test_km
        best_test_acc = test_acc

    scheduler.step()

    phase_label = "phase1" if epoch < PHASE1_EPOCHS else "phase2"
    print(
        f"Epoch {epoch+1:2d}/{FINAL_EPOCHS} [{phase_label} λ={lambda_cls}] | "
        f"Train: {train_km:.1f} km  acc {train_acc:.1f}% | "
        f"Test:  {test_km:.1f} km  acc {test_acc:.1f}%"
    )

print(f"\nMeilleur test, {best_test_km:.1f} km | acc pays {best_test_acc:.1f}%")

# ─────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────

MODEL_DIR = Path("models/resnet")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
save_path = MODEL_DIR / "resnet50_classif_regress.pt"

torch.save({
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epochs":               FINAL_EPOCHS,
    "best_test_km":         best_test_km,
    "best_test_acc":        best_test_acc,
    "country_to_idx":       dataset.country_to_idx,
    "config":               cfg,
}, save_path)

print(f"Modèle sauvegardé : {save_path}")
