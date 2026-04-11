"""
Géolocalisation d'images par régression GPS, Run final (hyperparamètres optimisés via Optuna)

Le modèle prend en entrée une photo de rue et prédit directement ses coordonnées GPS : c'est de la
régression pure sur les coordonnées encodées en sin/cos pour éviter les discontinuités
à ±180° de longitude et ±90° de latitude.

Pipeline :
    1. Le dataset charge les images depuis OSV5M et lit les coordonnées GPS dans un CSV.
       Les coordonnées (lat, lon) sont converties en 4 valeurs continues :
       [sin(lat), cos(lat), sin(lon), cos(lon)].

    2. Le backbone ResNet50 pré-entraîné sur ImageNet extrait les features visuelles.
       Toutes les couches sont dégelées (fine-tuning complet).
       La tête de classification originale est remplacée par un MLP → 4 sorties.
       Les sorties sont normalisées par paires (sin/cos) pour rester sur le cercle unité.

    3. La loss est la distance Haversine en km entre la coordonnée prédite et la vraie
       coordonnée, moyennée sur le batch. C'est une métrique directement interprétable.

    4. L'optimiseur est Adam avec un scheduler cosine annealing qui réduit le learning
       rate progressivement jusqu'à 0 sur les 30 epochs.

Hyperparamètres : issus du trial 23 d'une recherche Optuna sur 30 trials.
Dataset         : OSV5M (Open Street View 5M), 50 000 images pour ce run.
Résultat attendu : ~2000-2500 km de distance moyenne sur le jeu de test.

Usage :
    python train_final.py
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


# Dataset

class GeoDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, max_samples=None, seed=42):
        data = pd.read_csv(csv_file)
        if max_samples is not None:
            data = data.sample(
                n=min(max_samples, len(data)),
                random_state=seed
            ).reset_index(drop=True)
        self.data = data
        self.transform = transform

        self.image_index = {}
        for subdir in os.listdir(img_dir):
            subdir_path = os.path.join(img_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in os.listdir(subdir_path):
                if fname.endswith(".jpg"):
                    self.image_index[fname[:-4]] = os.path.join(subdir_path, fname)

        print(f"Images indexées : {len(self.image_index)}")
        print(f"Lignes CSV utilisées : {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = str(row["id"])

        if img_id not in self.image_index:
            raise FileNotFoundError(f"{img_id}.jpg absent du dataset")

        image = Image.open(self.image_index[img_id]).convert("RGB")

        lat = float(row["latitude"])
        lon = float(row["longitude"])
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        coords = torch.tensor([
            math.sin(lat_rad),
            math.cos(lat_rad),
            math.sin(lon_rad),
            math.cos(lon_rad),
        ], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, coords


# Haversine Loss

def sincos_to_rad(sin, cos):
    return torch.atan2(sin, cos)


class HaversineLoss(nn.Module):
    def __init__(self, radius=6371):
        super().__init__()
        self.radius = radius

    def forward(self, preds, targets):
        lat1 = sincos_to_rad(preds[:, 0], preds[:, 1])
        lon1 = sincos_to_rad(preds[:, 2], preds[:, 3])
        lat2 = sincos_to_rad(targets[:, 0], targets[:, 1])
        lon2 = sincos_to_rad(targets[:, 2], targets[:, 3])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2) ** 2 + \
            torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2

        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return (self.radius * c).mean()


# Modèle

class GeoResNet(nn.Module):
    def __init__(self, backbone="resnet50", hidden_dim=512, n_layers=1, dropout_p=0.4):
        super().__init__()

        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=True)

        in_features = self.backbone.fc.in_features

        if n_layers == 1:
            head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, 4),
            )
        else:
            head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim // 2, 4),
            )

        self.backbone.fc = head

    def forward(self, x):
        x = self.backbone(x)
        lat = F.normalize(x[:, 0:2], dim=1)
        lon = F.normalize(x[:, 2:4], dim=1)
        return torch.cat([lat, lon], dim=1)


# Meilleurs hyperparamètres (trial 23 Optuna)

BEST_PARAMS = {
    "backbone":       "resnet50",
    "hidden_dim":     512,
    "n_layers":       1,
    "dropout_p":      0.4,
    "lr":             0.00013083664196863518,
    "weight_decay":   0.00018965545687732138,
    "optimizer":      "adam",
    "scheduler":      "cosine",
    "unfreeze_layers": 3,
}

FINAL_EPOCHS = 30
MAX_SAMPLES  = 50000

# Setup device et batch size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    vram = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    BATCH_SIZE = 32 if vram >= 8 else 16  # 64 → 32

print(f"Device: {device} | Batch size: {BATCH_SIZE}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Chargement du dataset

dataset = GeoDataset(
    csv_file=os.path.expanduser("~/datasets/OSV5M/test_filtered.csv"),
    img_dir=os.path.expanduser("~/datasets/OSV5M"),
    transform=transform,
    max_samples=MAX_SAMPLES,
)

train_size = int(0.9 * len(dataset))
test_size  = len(dataset) - train_size
train_ds, test_ds = random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Initialisation du modèle et gel des couches

p = BEST_PARAMS

model = GeoResNet(
    backbone=p["backbone"],
    hidden_dim=p["hidden_dim"],
    n_layers=p["n_layers"],
    dropout_p=p["dropout_p"],
).to(device)

# Gel sélectif
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.backbone.fc.parameters():
    param.requires_grad = True
if p["unfreeze_layers"] >= 1:
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
if p["unfreeze_layers"] >= 2:
    for param in model.backbone.layer3.parameters():
        param.requires_grad = True
if p["unfreeze_layers"] >= 3:
    for param in model.backbone.parameters():
        param.requires_grad = True

# Optimiseur et scheduler

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=p["lr"],
    weight_decay=p["weight_decay"],
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

criterion = HaversineLoss()

# Boucle d'entraînement

best_test = float("inf")

print(f"\nEntraînement final, {FINAL_EPOCHS} epochs sur {MAX_SAMPLES} samples")

for epoch in range(FINAL_EPOCHS):

    # Train
    model.train()
    train_total = 0.0
    for imgs, coords in train_loader:
        imgs, coords = imgs.to(device), coords.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, coords)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_total += loss.item()

    # Eval
    model.eval()
    test_total = 0.0
    with torch.no_grad():
        for imgs, coords in test_loader:
            imgs, coords = imgs.to(device), coords.to(device)
            test_total += criterion(model(imgs), coords).item()

    train_loss = train_total / len(train_loader)
    test_loss  = test_total  / len(test_loader)

    if test_loss < best_test:
        best_test = test_loss

    scheduler.step()

    print(f"Epoch {epoch+1:2d}/{FINAL_EPOCHS} | Train: {train_loss:.2f} km | Test: {test_loss:.2f} km")

print(f"\nMeilleur test loss : {best_test:.2f} km")

# Sauvegarde

MODEL_DIR = Path("models/resnet")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
save_path = MODEL_DIR / "resnet50_geo_final.pt"

torch.save({
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epochs":               FINAL_EPOCHS,
    "best_test_km":         best_test,
    "hyperparams":          p,
}, save_path)

print(f"Modèle sauvegardé : {save_path}")