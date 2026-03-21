"""
1. Charger resnet50(pretrained=True)
2. Accéder aux blocs Bottleneck existants
3. "Wrapper" chaque bloc avec CBAM
4. Les poids CNN restent pretrained
5. Les poids CBAM s'initialisent aléatoirement et s'apprennent

Utilisation :
torchvision.models.resnet50(pretrained=True) = même architecture 
mais avec les poids ImageNet pré-entraînés. C'est du transfer learning
(le réseau a déjà appris à détecter des textures, bords, formes génériques 
sur 1.2M d'images).


ResNet 50 :
class GeoGuessrModel(nn.Module):
    def __init__(self, num_countries):
        - charger resnet50_cbam()
        - stocker les couches du backbone sans fc
        - définir tête pays  (2048 → ... → num_countries)
        - définir tête GPS   (2048 → ... → 2) + tanh + dénorm

    def forward(self, x):
        - passer par chaque couche backbone manuellement
        - flatten
        - tête pays → softmax
        - tête GPS  → tanh → dénorm
        - retourner (pred_pays, pred_GPS)

loss_totale = CrossEntropy(pred_pays, label_pays) 
            + λ * Haversine(pred_GPS, label_GPS)


Normalisation des coordonnées GPS résumé complet

Pourquoi normaliser ?

Les réseaux de neurones fonctionnent mal avec des grandes valeurs brutes. Les poids sont initialisés proches de 0, les activations aussi — des valeurs comme `48.85` ou `139.7` créent des gradients instables.

lat  ∈ [-90,  +90]   → trop grand pour le réseau
lon  ∈ [-180, +180]  → trop grand pour le réseau

La normalisation simple — diviser par le max

lat_norm = lat / 90    → [-1, +1]
lon_norm = lon / 180   → [-1, +1]

Et la dénormalisation inverse :

lat = lat_norm * 90
lon = lon_norm * 180

Tanh l'activation finale de la tête GPS

La tête GPS doit sortir des valeurs entre -1 et +1. Tanh fait exactement ça :

tanh(x) → ]-1, +1[

Pas Sigmoid qui sort entre 0 et +1.

Le pipeline complet

ENTRAÎNEMENT :

Labels dans le dataset
lat=48.85, lon=2.35 (degrés bruts)
          ↓
         pas de normalisation des labels
         on garde les degrés bruts comme target


Forward pass :
Image → backbone → features [2048]
                       ↓
                   tête GPS
                       ↓
               Linear → ... → Linear
                       ↓
                     tanh          → sortie ∈ [-1, +1]
                       ↓
                  * [90, 180]      → degrés réels ∈ [-90/+90, -180/+180]
                       ↓
              haversine_loss(pred_degrés, target_degrés)

Pourquoi dénormaliser dans le forward et pas dans la loss ?

Parce que la Haversine loss attend des **degrés réels** — elle fait des `torch.deg2rad()` en interne. Si tu lui passes des valeurs entre -1 et +1, elle calcule une distance absurde.

Dans le forward :
out = self.linear_gps(x)      # valeurs quelconques
out = torch.tanh(out)          # → [-1, +1]
out = out * torch.tensor([90., 180.])  # → degrés réels
return out

Dans la loss :
haversine_loss(pred, target)   # pred ET target en degrés ✅

Le problème du méridien 180° — pourquoi 3 sorties en V2

Avec 2 sorties `(lat, lon)` :

Point A : lon = -179°  →  lon_norm = -0.994
Point B : lon = +179°  →  lon_norm = +0.994

Distance réelle    : ~2km   (ils sont voisins dans le Pacifique)
Distance pour MSE  : |−0.994 − 0.994| = 1.988  → énorme pénalité 


Avec 3 sorties cartésiennes `(x, y, z)` :

x = cos(lat) * cos(lon)
y = cos(lat) * sin(lon)
z = sin(lat)

Point A et Point B → coordonnées (x,y,z) très proches 

Mais pour la V1 — 2 sorties suffisent. Le méridien 180° concerne une minorité d'images dans OSV5M. Tu passes à 3 sorties si tu vois des erreurs aberrantes sur le Pacifique.

Récapitulatif visuel

                    Réseau
                 ┌──────────┐
lat=48.85° ───→  │          │
lon=2.35°  ───→  │ backbone │──→ tanh ──→ *[90,180] ──→ 48.72°, 2.41°
(target)         │  + CBAM  │                              ↓
                 └──────────┘                     haversine_loss
                                                  (km entre pred et target)


Les valeurs importantes à retenir

| Variable | Plage brute | Plage normalisée | Facteur |
|---|---|---|---|
| Latitude | [-90, +90] | [-1, +1] via tanh | ×90 |
| Longitude | [-180, +180] | [-1, +1] via tanh | ×180 |

Image Street View
      ↓
conv1 → bn1 → relu → maxpool
      ↓
layer1 [3 × BottleneckWithCBAM]
layer2 [4 × BottleneckWithCBAM]
layer3 [6 × BottleneckWithCBAM]
layer4 [3 × BottleneckWithCBAM]
      ↓
avgpool → flatten [2048]
      ↙               ↘
head_countries      head_gps
Linear 2048→512     Linear 2048→512
ReLU                ReLU
Dropout             Dropout
Linear 512→N        Linear 512→2
                    Tanh → *[90,180]
      ↓               ↓
pred_countries    pred_gps (lat, lon)


Phase 1 — Têtes seulement
backbone  →  gelé       (requires_grad = False)
têtes     →  apprennent (requires_grad = True)

But : stabiliser les têtes avant de toucher au backbone
Durée : quelques epochs (5-10)

          ↓

Phase 2 — Fine-tuning complet
backbone  →  dégelé, lr faible  (1e-4)
têtes     →  continuent, lr faible (1e-4)

But : adapter le backbone à GeoGuessr tout en gardant
      ce qu'il a appris sur ImageNet
Durée : plus long (20-50 epochs)


## Pourquoi des lr différents ?

### Le backbone — pretrained sur ImageNet

Il a déjà appris des **features génériques très utiles** — bords, textures, formes, végétation. Ces features sont précieuses et fragiles.

Un grand lr les **écrase** :

```
gradient × lr_grand  →  mise à jour trop grande
→ les features ImageNet sont détruites
→ tu repars de zéro sans le savoir
```

Un petit lr les **préserve et affine doucement** :

```
gradient × lr_petit  →  petite mise à jour
→ les features s'adaptent progressivement à GeoGuessr
→ tu gardes le bénéfice du pretraining 
```

---

### Les têtes — initialisées aléatoirement

Elles partent de zéro — elles ont besoin d'apprendre **vite** pour converger :

```
lr grand  →  apprentissage rapide 
lr petit  →  convergence très lente, entraînement inefficace ❌
```

---

### Visuellement

```
ImageNet features          Tâche GeoGuessr
     ↓                           ↓
backbone lr=1e-4           têtes lr=1e-3
"je peaufine"              "j'apprends"
petits pas                 grands pas
```

---

### L'analogie

C'est comme rénover une maison ancienne :

**Backbone** = murs porteurs → on touche avec précaution, petits ajustements

**Têtes** = décoration intérieure → on peut tout refaire rapidement sans risque

En phase 1 les têtes ont appris avec `lr=1e-3` — elles sont **déjà partiellement convergées**.

Si tu gardes `lr=1e-3` en phase 2 :
```
têtes déjà convergées + lr encore grand
→ elles oscillent autour du minimum au lieu de converger
→ instabilité qui se propage dans le backbone via les gradients 
```

En baissant à `lr=1e-4` en phase 2 :
```
backbone    lr=1e-4  → affinage doux du pretrained 
têtes       lr=1e-4  → affinage doux des têtes convergées 
tout le réseau évolue lentement et stablement 
"""

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from cbam import CBAM
import joblib
import os                   
import numpy as np                
import pandas as pd                
from PIL import Image              
from sklearn.preprocessing import LabelEncoder   
from sklearn.metrics import f1_score            
from torch.utils.data import DataLoader, random_split 
import random
from torch.utils.data import Subset


def haversine_loss(pred, target, epsSq=1.e-13, epsAs=1.e-7):
    lat1, lon1 = torch.split(pred, 1, dim=1)
    lat2, lon2 = torch.split(target, 1, dim=1)
    r = 6371
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi = torch.deg2rad(lat2 - lat1)
    delta_lambda = torch.deg2rad(lon2 - lon1)
    a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
    return torch.Tensor.mean(2 * r * torch.asin((1.0 - epsAs) * torch.sqrt(a + (1.0 - a**2) * epsSq)))



class BottleneckWithCBAM(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
        self.bottleneck = bottleneck
        planes = bottleneck.conv3.out_channels
        self.cbam = CBAM(planes, 16)

    def forward(self,x):
        return self.cbam(self.bottleneck(x))


def resnet50_cbam(**kwargs):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4] :
        for i in range(len(layer)):
            layer[i] = BottleneckWithCBAM(layer[i]) 

    return model


class GeoGussrAttentionMultiTask(nn.Module):
    def __init__(self, num_countries):
        super().__init__()
        self.num_countries = num_countries
        model = resnet50_cbam()
        for name, layer in model.named_children():
            if name != "fc":
                setattr(self, name, layer)
                
        self.head_countries = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_countries)
        )
        self.head_gps = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        pred_countries = self.head_countries(x)
        pred_gps = self.head_gps(x)
        pred_gps = pred_gps * torch.tensor([90.,180.], device=x.device)
        return pred_countries, pred_gps


class GeoGuesserIADataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.le = LabelEncoder()
        self.le.fit(self.df['country'])
        self.transform = transform

        self.image_index = {}
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in os.listdir(subdir_path):
                if fname.endswith(".jpg"):
                    self.image_index[fname[:-4]] = os.path.join(subdir_path, fname)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_id = str(self.df.iloc[idx]['id'])
        img = Image.open(self.image_index[img_id])

        if self.transform :
            img =self.transform(img)
        
        country = self.df.iloc[idx]['country']
        label_country = self.le.transform([country])[0]
        label_country = torch.tensor(label_country, dtype=torch.long)

        lat = self.df.iloc[idx]['latitude']
        long = self.df.iloc[idx]['longitude']
        label_gps = torch.tensor([lat, long], dtype=torch.float32)

        return img, label_country, label_gps

batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


train_dataset = GeoGuesserIADataset(
    csv_path='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/test_filtered.csv',
    root_dir='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/images/test',
    transform=train_transform
)

val_test_dataset = GeoGuesserIADataset(
    csv_path='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/test_filtered.csv',
    root_dir='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/images/test',
    transform=val_test_transform
)

total = len(train_dataset)
indices = list(range(total))

train_size = int(0.8 * total)
val_size = int(0.1 * total)
test_size = total - train_size - val_size

random.shuffle(indices)

train_indices    = indices[:train_size]
val_indices      = indices[train_size:train_size + val_size]
test_indices     = indices[train_size + val_size:]

train_dataset_final = Subset(train_dataset,    train_indices)
val_dataset_final   = Subset(val_test_dataset, val_indices)
test_dataset_final  = Subset(val_test_dataset, test_indices)


train_loader = DataLoader(
    train_dataset_final, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4, 
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset_final, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=4, 
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset_final, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=4, 
    pin_memory=True
)

dataloader = {
    'train' : train_loader,
    'test' : test_loader,
    'validation' : val_loader
}

num_countries = len(train_dataset.le.classes_)
model = GeoGussrAttentionMultiTask(num_countries).to(device)

criterion_countries = nn.CrossEntropyLoss()
criterion_gps = haversine_loss

head_params_ids = set(
    id(p) for p in list(model.head_countries.parameters()) + list(model.head_gps.parameters())
)

backbone_params = [p for p in model.parameters() if id(p) not in head_params_ids]

optimizer = optim.Adam([
    {'params': backbone_params, 'lr':1e-4},
    {'params': model.head_countries.parameters(), 'lr':1e-3},
    {'params': model.head_gps.parameters(), 'lr':1e-3}
])

lambda_gps = 1.0

NUM_EPOCH_PHASE1 = 5
NUM_EPOCH_PHASE2 = 20

def train_model(model, optimizer, num_epochs=NUM_EPOCH_PHASE1 + NUM_EPOCH_PHASE2):
    
    for name, module in model.named_children():
        if name not in ['head_countries', 'head_gps']:
            for param in module.parameters():
                param.requires_grad = False
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        if epoch == NUM_EPOCH_PHASE1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer.param_groups[0]['lr'] = 1e-4
            optimizer.param_groups[1]['lr'] = 1e-4 
            optimizer.param_groups[2]['lr'] = 1e-4

        for phase in ['train', 'validation']:
            running_loss_country = 0.0   
            running_loss_gps = 0.0  
            running_loss_total = 0.0   
            
            all_preds = []
            all_labels = []

            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            for inputs, labels_country, labels_gps in dataloader[phase]:
                inputs = inputs.to(device)
                labels_country = labels_country.to(device)
                labels_gps = labels_gps.to(device)

                if phase == 'validation':
                    with torch.no_grad():
                        pred_countries, pred_gps = model(inputs)
                else:
                    optimizer.zero_grad()
                    pred_countries, pred_gps = model(inputs)
                
                all_preds.append(torch.argmax(pred_countries, dim=1).detach().cpu().numpy())
                all_labels.append(labels_country.detach().cpu().numpy())

                loss_country = criterion_countries(pred_countries, labels_country)
                loss_gps = haversine_loss(pred_gps, labels_gps)
                loss_gps_normalized = loss_gps / 20000
                loss = loss_country + lambda_gps * loss_gps_normalized

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss_country += loss_country.detach() * inputs.size(0)
                running_loss_gps += loss_gps.detach() * inputs.size(0)
                running_loss_total += loss.detach() * inputs.size(0)
                
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            f1 = f1_score(all_labels, all_preds, average='macro')
            
            epoch_loss_country = running_loss_country / len(dataloader[phase].dataset)
            epoch_loss_total = running_loss_total / len(dataloader[phase].dataset)
            epoch_gps_dist = running_loss_gps / len(dataloader[phase].dataset)
            
            print('{} | loss_total: {:.4f} | loss_country: {:.4f} | loss_gps: {:.1f}km | f1: {:.4f}'.format(
                    phase,
                    epoch_loss_total,
                    epoch_loss_country,
                    epoch_gps_dist,
                    f1
                ))
    return model

if __name__ == '__main__':
    model_trained = train_model(model, optimizer)
    torch.save(model_trained.state_dict(), 'geoguessr_model_attention_classif_reg.pt')
    joblib.dump(train_dataset.le, 'label_encoder.pkl')
    print('Modèle sauvegardé')




