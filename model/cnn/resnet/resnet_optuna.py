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
import optuna
from optuna.trial import TrialState
import logging

# Silencer les warnings torchvision
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Dataset
# -----------------------------

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


# -----------------------------
# Haversine Loss
# -----------------------------

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


# -----------------------------
# Modèle paramétrable
# -----------------------------

class GeoResNet(nn.Module):
    """
    Backbone ResNet18 ou ResNet50 avec tête MLP configurable.
    n_layers : nombre de couches cachées (1 ou 2)
    hidden_dim : taille de la couche cachée
    dropout_p : taux de dropout
    """

    def __init__(self, backbone="resnet18", hidden_dim=512, n_layers=1, dropout_p=0.3):
        super().__init__()

        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Backbone inconnu : {backbone}")

        in_features = self.backbone.fc.in_features

        if n_layers == 1:
            head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, 4),
            )
        else:  # 2 couches cachées
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


# -----------------------------
# Config globale
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Dataset partagé entre tous les trials (chargé une seule fois)
BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# On charge le dataset une seule fois — trials partagent le même split
_dataset = GeoDataset(
    csv_file=os.path.expanduser("~/datasets/OSV5M/test_filtered.csv"),
    img_dir=os.path.expanduser("~/datasets/OSV5M"),
    transform=BASE_TRANSFORM,
    # Réduit pour que les trials soient rapides ; augmenter pour le run final
    max_samples=10000,
)

_train_size = int(0.9 * len(_dataset))
_test_size = len(_dataset) - _train_size
_train_ds, _test_ds = random_split(
    _dataset, [_train_size, _test_size],
    generator=torch.Generator().manual_seed(42)
)

# Batch size fixe basé sur la VRAM disponible
if device.type == "cuda":
    vram = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    BATCH_SIZE = 64 if vram >= 8 else 32
else:
    BATCH_SIZE = 16

# Nombre d'epochs par trial (court pour HPO, augmenter pour final)
OPTUNA_EPOCHS = 8


# -----------------------------
# Objective Optuna
# -----------------------------

def objective(trial: optuna.Trial) -> float:
    # --- Hyperparamètres à tuner ---

    # Backbone
    backbone = trial.suggest_categorical("backbone", ["resnet18", "resnet50"])

    # Tête MLP
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    n_layers = trial.suggest_int("n_layers", 1, 2)
    dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5, step=0.05)

    # Optimiseur
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])

    # Scheduler
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["none", "cosine", "reduce_on_plateau"]
    )

    # Fine-tuning : dégeler les derniers blocs du backbone
    unfreeze_layers = trial.suggest_int("unfreeze_layers", 0, 3)
    # 0 = tête seulement, 1 = layer4, 2 = layer3+4, 3 = tout

    # --- Construction du modèle ---
    model = GeoResNet(
        backbone=backbone,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout_p=dropout_p,
    ).to(device)

    # Gel sélectif
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Toujours dégeler la tête
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    if unfreeze_layers >= 1:
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
    if unfreeze_layers >= 2:
        for param in model.backbone.layer3.parameters():
            param.requires_grad = True
    if unfreeze_layers >= 3:
        for param in model.backbone.parameters():
            param.requires_grad = True

    # --- Optimiseur ---
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # --- Scheduler ---
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=OPTUNA_EPOCHS
        )
    elif scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
    else:
        scheduler = None

    # --- DataLoaders ---
    train_loader = DataLoader(
        _train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        _test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    criterion = HaversineLoss()
    best_val = float("inf")

    for epoch in range(OPTUNA_EPOCHS):

        # --- Train ---
        model.train()
        train_total = 0.0
        for imgs, coords in train_loader:
            imgs, coords = imgs.to(device), coords.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, coords)
            loss.backward()
            # Gradient clipping pour la stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += loss.item()

        train_loss = train_total / len(train_loader)

        # --- Eval ---
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for imgs, coords in val_loader:
                imgs, coords = imgs.to(device), coords.to(device)
                preds = model(imgs)
                val_total += criterion(preds, coords).item()

        val_loss = val_total / len(val_loader)

        if val_loss < best_val:
            best_val = val_loss

        # Scheduler step
        if scheduler is not None:
            if scheduler_name == "reduce_on_plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        logger.info(
            f"  Trial {trial.number} | Epoch {epoch+1}/{OPTUNA_EPOCHS} "
            f"| Train {train_loss:.0f} km | Val {val_loss:.0f} km"
        )

        # Pruning Optuna (arrête les trials peu prometteurs)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val


# -----------------------------
# Lancer l'étude
# -----------------------------

if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize",
        study_name="geo_resnet_hpo",
        # Stockage persistant : reprendre l'étude si interruption
        storage="sqlite:///optuna_geo.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=3,
        ),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Nombre de trials HPO — augmenter selon le temps disponible
    N_TRIALS = 30

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=None,           # ou mettre un timeout en secondes
        gc_after_trial=True,    # libère la mémoire GPU entre les trials
    )

    # -----------------------------
    # Résultats
    # -----------------------------

    print("\n" + "=" * 60)
    print("OPTUNA — Résultats")
    print("=" * 60)

    pruned = len(study.get_trials(states=[TrialState.PRUNED]))
    complete = len(study.get_trials(states=[TrialState.COMPLETE]))
    print(f"Trials terminés : {complete} | Pruned : {pruned}")

    best = study.best_trial
    print(f"\nMeilleur trial  : #{best.number}")
    print(f"Meilleure val   : {best.value:.2f} km")
    print("\nHyperparamètres optimaux :")
    for k, v in best.params.items():
        print(f"  {k:25s} = {v}")

    # -----------------------------
    # Ré-entraînement final avec les meilleurs HP
    # -----------------------------

    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT FINAL avec les meilleurs hyperparamètres")
    print("=" * 60)

    p = best.params

    # Dataset complet pour le run final
    full_dataset = GeoDataset(
        csv_file=os.path.expanduser("~/datasets/OSV5M/test_filtered.csv"),
        img_dir=os.path.expanduser("~/datasets/OSV5M"),
        transform=BASE_TRANSFORM,
        max_samples=50000,      # Remettre le max comme ton run original
    )

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = GeoResNet(
        backbone=p["backbone"],
        hidden_dim=p["hidden_dim"],
        n_layers=p["n_layers"],
        dropout_p=p["dropout_p"],
    ).to(device)

    # Appliquer le même gel/dégel
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

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if p["optimizer"] == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=p["lr"], weight_decay=p["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=p["lr"], weight_decay=p["weight_decay"])

    FINAL_EPOCHS = 30
    if p["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)
    elif p["scheduler"] == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    else:
        scheduler = None

    criterion = HaversineLoss()
    best_test = float("inf")

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
        test_loss = test_total / len(test_loader)

        if test_loss < best_test:
            best_test = test_loss

        if scheduler is not None:
            if p["scheduler"] == "reduce_on_plateau":
                scheduler.step(test_loss)
            else:
                scheduler.step()

        print(f"Epoch {epoch+1}/{FINAL_EPOCHS} | Train: {train_loss:.2f} km | Test: {test_loss:.2f} km")

    print(f"\nMeilleur test loss final : {best_test:.2f} km")

    # Save
    MODEL_DIR = Path("models/resnet")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / "resnet_geo_optuna.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs": FINAL_EPOCHS,
        "best_val_km": best_test,
        "hyperparams": p,
    }, save_path)
    print(f"Modèle sauvegardé : {save_path}")