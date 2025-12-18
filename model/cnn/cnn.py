import json
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# === ÉTAPE 1 : Préparer les données et créer les JSON ===

json_path = "/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_kaggle/label_association/dataset_metadata_kaggle.json"
with open(json_path, "r") as f:
    metadata = json.load(f)

for item in metadata["images"]:
    item["path"] = item["path"].replace("\\", "/")

images_list = metadata["images"]
image_paths = [item["path"] for item in images_list]
labels = [item["country"] for item in images_list]

with open(json_path, "w") as f:
    json.dump(metadata, f, indent=2)

# Split stratifié
train_paths, test_paths = train_test_split(
    image_paths, 
    test_size=0.2, 
    random_state=42, 
    #stratify=labels
)

# Créer les deux JSON
train_items = [item for item in images_list if item["path"] in train_paths]
test_items = [item for item in images_list if item["path"] in test_paths]

with open("train_metadata.json", "w") as f:
    json.dump({"images": train_items}, f)

with open("test_metadata.json", "w") as f:
    json.dump({"images": test_items}, f)

print(f"Train : {len(train_items)} images")
print(f"Test : {len(test_items)} images")


# === ÉTAPE 2 : Créer le Dataset personnalisé ===

class GeoDataset(Dataset):
    def __init__(self, image_dir, json_path, transform=None, label_to_idx=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Charger le JSON
        with open(json_path, "r") as f:
            metadata = json.load(f)
        
        self.images_list = metadata["images"]
        
        # Utiliser le mapping fourni ou en créer un nouveau
        if label_to_idx is None:
            unique_labels = sorted(list(set([item["country"] for item in self.images_list])))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
        
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        item = self.images_list[idx]
        
        # Charger l'image
        img_path = os.path.join(self.image_dir, item["path"])
        image = Image.open(img_path).convert("RGB")
        
        # Obtenir le label numérique
        label = self.label_to_idx[item["country"]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# === ÉTAPE 3 : Définir les transformations ===

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])


# === ÉTAPE 4 : Créer un mapping global pour tous les labels ===

# Créer le mapping à partir de TOUTES les images (train + test)
unique_labels = sorted(list(set([item["country"] for item in images_list])))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(unique_labels)

print(f"Nombre total de classes : {num_classes}")

root = "/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset/compressed_dataset/"

# Passer le même mapping aux deux datasets
train_dataset = GeoDataset(
    image_dir=root,
    json_path="/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/train_metadata.json",
    transform=transform,
    label_to_idx=label_to_idx  # ← Utiliser le mapping global
)

test_dataset = GeoDataset(
    image_dir=root,
    json_path="/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/test_metadata.json",
    transform=transform,
    label_to_idx=label_to_idx  # ← Utiliser le mapping global
)


# === ÉTAPE 5 : Créer les DataLoaders ===

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# === ÉTAPE 6 : Créer le modèle ===

# Charger ResNet18 pré-entraîné (utiliser weights au lieu de pretrained)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Modifier la dernière couche
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Optimizer et loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# === ÉTAPE 7 : Fonctions d'entraînement et d'évaluation ===

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(train_loader), correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return loss / len(test_loader), correct / total


# === ÉTAPE 8 : Entraînement ===

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)

for epoch in range(10):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/10")
    print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
    print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")


# === ÉTAPE 9 : Évaluation finale ===

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculer l'accuracy
accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"\nFinal Accuracy: {accuracy:.4f}")


# === ÉTAPE 10 : Visualisation et métriques ===

# Trouver les classes qui apparaissent réellement dans le test set
classes_in_test = sorted(list(set(all_labels)))
labels_for_report = [idx_to_label[idx] for idx in classes_in_test]

print(f"\nNombre de classes dans le test set: {len(classes_in_test)} sur {num_classes}")

# Matrice de confusion (seulement pour les classes présentes)
cm = confusion_matrix(all_labels, all_preds, labels=classes_in_test)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
            xticklabels=labels_for_report, yticklabels=labels_for_report)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Rapport détaillé
print("\n" + "="*50)
print("Classification Report")
print("="*50)
print(classification_report(all_labels, all_preds, 
                          labels=classes_in_test,
                          target_names=labels_for_report))

# Sauvegarder le modèle
torch.save({
    'model_state_dict': model.state_dict(),
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'num_classes': num_classes
}, 'geoguesser_model.pth')
print("\nModèle sauvegardé dans 'geoguesser_model.pth'")