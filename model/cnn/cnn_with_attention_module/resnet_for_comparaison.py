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
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import random
from torch.utils.data import Subset
import math


def haversine_loss(pred, target, epsSq=1.e-13, epsAs=1.e-7):
    lat1, lon1 = torch.split(pred, 1, dim=1)
    lat2, lon2 = torch.split(target, 1, dim=1)
    r = 6371
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi = torch.deg2rad(lat2 - lat1)
    delta_lambda = torch.deg2rad(lon2 - lon1)
    a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
    return torch.Tensor.mean(2 * r * torch.asin((1.0 - epsAs) * torch.sqrt(a + (1.0 - a**2) * epsSq)))

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
    def __init__(self, num_countries, embed_detach=False):
        super().__init__()
        self.num_countries = num_countries
        self.embed_detach = embed_detach
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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
            nn.Linear(2048 + num_countries, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 4)
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
        feats = x.flatten(1)                             
        pred_countries = self.head_countries(feats)       
        cls_probs = torch.softmax(pred_countries, dim=1)
        embed = cls_probs.detach() if self.embed_detach else cls_probs
        reg_input = torch.cat([feats, embed], dim=1)
        raw = self.head_gps(reg_input)
        lat = F.normalize(raw[:, 0:2], dim=1)
        lon = F.normalize(raw[:, 2:4], dim=1)
        pred_gps = torch.cat([lat, lon], dim=1)

        return pred_countries, pred_gps


class GeoGuesserIADataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path, low_memory=False)
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

        lat_r = math.radians(float(self.df.iloc[idx]['latitude']))   
        lon_r = math.radians(float(self.df.iloc[idx]['longitude']))  
        label_gps = torch.tensor(
            [math.sin(lat_r), math.cos(lat_r), math.sin(lon_r), math.cos(lon_r)],
            dtype=torch.float32,
        )

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
    csv_path='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/samples_filtered_v2.csv',
    root_dir='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/images/samples',
    transform=train_transform
)

val_test_dataset = GeoGuesserIADataset(
    csv_path='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/samples_filtered_v2.csv',
    root_dir='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/images/samples',
    transform=val_test_transform
)

total = len(train_dataset)
indices = list(range(total))

train_size = int(0.8 * total)
val_size = int(0.1 * total)
test_size = total - train_size - val_size

random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_dataset_final = Subset(train_dataset,    train_indices)
val_dataset_final = Subset(val_test_dataset, val_indices)
test_dataset_final = Subset(val_test_dataset, test_indices)


train_countries = [train_dataset.df.iloc[i]['country'] for i in train_indices]
country_counts  = pd.Series(train_countries).value_counts().to_dict()
total_train     = len(train_indices)
n_classes       = len(country_counts)

sample_weights = torch.tensor([
    total_train / (n_classes * country_counts[c])
    for c in train_countries
], dtype=torch.float32)

sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(sample_weights),
    replacement = True,
)


train_loader = DataLoader(
    train_dataset_final, 
    batch_size=batch_size, 
    shuffle=False,
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
model = GeoGussrAttentionMultiTask(num_countries, embed_detach=True).to(device)

criterion_countries = nn.CrossEntropyLoss()
criterion_gps = HaversineLoss()

head_params_ids = set(
    id(p) for p in list(model.head_countries.parameters()) + list(model.head_gps.parameters())
)

backbone_params = [p for p in model.parameters() if id(p) not in head_params_ids]

optimizer = optim.Adam([
    {'params': backbone_params, 'lr':1e-4},
    {'params': model.head_countries.parameters(), 'lr':1e-3},
    {'params': model.head_gps.parameters(), 'lr':1e-3}
])

LAMBDA_REG = 0.0001 

NUM_EPOCH_PHASE1 = 10
NUM_EPOCH_PHASE2 = 45

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
            optimizer.param_groups[0]['lr'] = 1e-5
            optimizer.param_groups[1]['lr'] = 1e-5 
            optimizer.param_groups[2]['lr'] = 1e-5

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
                loss_gps =  criterion_gps(pred_gps, labels_gps)

                if epoch < NUM_EPOCH_PHASE1:
                    loss = loss_country + 0.00001 * loss_gps
                else:
                    warmup = min((epoch - NUM_EPOCH_PHASE1) / 5.0, 1.0)
                    loss = loss_country + (LAMBDA_REG * warmup) * loss_gps

                if phase == 'train':
                    if torch.isnan(loss) or torch.isinf(loss):
                        optimizer.zero_grad()
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
    torch.save(model_trained.state_dict(), 'geoguessr_model_classif_comparaison_samples.pt')
    joblib.dump(train_dataset.le, 'label_encoder_comparaison_samples.pkl')
    print('Modèle sauvegardé')




