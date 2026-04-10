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
from resnet_cbam import haversine_loss, sincos_to_rad, HaversineLoss, BottleneckWithCBAM, resnet50_cbam, GeoGussrAttentionMultiTask, GeoGuesserIADataset


batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomVerticalFlip(),
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
    csv_path='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/rest_filtered_v2.csv',
    root_dir='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/images/rest_images',
    transform=train_transform
)

val_test_dataset = GeoGuesserIADataset(
    csv_path='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/rest_filtered_v2.csv',
    root_dir='/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/images/rest_images',
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


# Undersampling US
US_CAP = 30_000

train_countries_list = [train_dataset.df.iloc[i]['country'] for i in train_indices]

us_positions     = [i for i, c in enumerate(train_countries_list) if c == 'US']
non_us_positions = [i for i, c in enumerate(train_countries_list) if c != 'US']

random.seed(42)
us_positions_kept = random.sample(us_positions, min(US_CAP, len(us_positions)))

balanced_positions = sorted(us_positions_kept + non_us_positions)
train_indices_balanced = [train_indices[p] for p in balanced_positions]

train_dataset_final = Subset(train_dataset, train_indices_balanced)

print(f"Train avant : {len(train_indices):,} après undersampling US : {len(train_indices_balanced):,}")

train_loader = DataLoader(
    train_dataset_final,
    batch_size  = batch_size,
    shuffle     = True,
    num_workers = 8,
    pin_memory  = True,
)

test_loader = DataLoader(
    test_dataset_final, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=8, 
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset_final, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=8, 
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
    torch.save(model_trained.state_dict(), 'geoguessr_model_attention_classif_UndersamplingUS_rest.pt')
    joblib.dump(train_dataset.le, 'label_encoder_UndersamplingUS_rest.pkl')
    print('Modèle sauvegardé')




