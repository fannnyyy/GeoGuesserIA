import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd

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
# Transforms avec data augmentation
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Device & batch size auto
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if vram < 4:
        batch_size = 16
    elif vram < 8:
        batch_size = 32
    elif vram < 12:
        batch_size = 64
    elif vram < 16:
        batch_size = 128
    else:
        batch_size = 256
else:
    batch_size = 16

print(f"Device: {device}, batch size: {batch_size}")

# -----------------------------
# Dataset & loaders
# -----------------------------
dataset = GeoDataset(
    csv_file=os.path.expanduser("~/datasets/OSV5M/test_filtered.csv"),
    img_dir=os.path.expanduser("~/datasets/OSV5M"),
    transform=transform,
    max_samples=20000
)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# -----------------------------
# CNN amélioré
# -----------------------------
class ImprovedCNN(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()

        # Bloc 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Bloc 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Bloc 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Bloc 4
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_p)

        # Fully connected layer réduit
        self.fc1 = nn.Linear(128*14*14, 256)
        self.fc2 = nn.Linear(256, 4)  # sin/cos lat/lon

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # normalisation sin/cos
        lat = F.normalize(x[:, 0:2], dim=1)
        lon = F.normalize(x[:, 2:4], dim=1)
        return torch.cat([lat, lon], dim=1)

# -----------------------------
# Haversine loss
# -----------------------------
def sincos_to_rad(sin, cos):
    return torch.atan2(sin, cos)

class HaversineLoss(nn.Module):
    def __init__(self, radius=6371):
        super().__init__()
        self.radius = radius

    def forward(self, preds, targets):
        lat1 = sincos_to_rad(preds[:,0], preds[:,1])
        lon1 = sincos_to_rad(preds[:,2], preds[:,3])
        lat2 = sincos_to_rad(targets[:,0], targets[:,1])
        lon2 = sincos_to_rad(targets[:,2], targets[:,3])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2)**2 + \
            torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2

        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return (self.radius * c).mean()

# -----------------------------
# Training utils
# -----------------------------
model = ImprovedCNN().to(device)
criterion = HaversineLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_epoch(model, loader):
    model.train()
    total = 0

    for imgs, coords in loader:
        imgs = imgs.to(device)
        coords = coords.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, coords)
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)

def eval_model(model, loader):
    model.eval()
    total = 0

    with torch.no_grad():
        for imgs, coords in loader:
            imgs = imgs.to(device)
            coords = coords.to(device)
            preds = model(imgs)
            total += criterion(preds, coords).item()

    return total / len(loader)

# -----------------------------
# Training loop
# -----------------------------
epochs = 20

for e in range(epochs):
    train_loss = train_epoch(model, train_loader)
    test_loss = eval_model(model, test_loader)

    print(
        f"Epoch {e+1}/{epochs} | "
        f"Train: {train_loss:.2f} km | "
        f"Test: {test_loss:.2f} km"
    )

# -----------------------------
# Save model
# -----------------------------
MODEL_DIR = Path("models/cnn")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

save_path = MODEL_DIR / "cnn_final.pt"

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epochs": epochs,
    "batch_size": batch_size,
    "loss": "haversine_sincos"
}, save_path)

print(f"Modèle sauvegardé dans {save_path}")
