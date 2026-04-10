## Comparaison détaillée — ton code vs code collègue

---

### 1. Architecture du backbone

**Ton code (CBAM)**
```python
model = resnet50_cbam()  # ResNet50 + CBAM dans chaque Bottleneck
for name, layer in model.named_children():
    if name != "fc":
        setattr(self, name, layer)  # copie couche par couche
```
Les couches sont copiées comme attributs directs — `self.conv1`, `self.layer1`... Ce pattern permet d'accéder directement à `model.layer4[-1].cbam` pour GradCAM mais rend l'introspection moins standard.

**Code collègue**
```python
backbone = models.resnet50(pretrained=True)
backbone.fc = nn.Identity()
self.backbone = backbone  # backbone encapsulé dans un seul attribut
```
Backbone encapsulé proprement dans `self.backbone` — accès via `model.backbone.layer4`. Plus standard, plus compatible avec `pytorch_grad_cam` sans wrapper.

---

### 2. Modules d'attention

**Ton code** — CBAM dans chaque bloc Bottleneck :
```python
layer[i] = BottleneckWithCBAM(layer[i])
# → Channel Attention + Spatial Attention après chaque bloc résiduel
```

**Code collègue** — aucun module d'attention. ResNet50 standard pretrained.

C'est la différence fondamentale du projet — ton modèle teste si CBAM améliore la discrimination géographique.

---

### 3. Têtes de prédiction

**Ton code**
```python
self.head_countries = nn.Sequential(Linear(2048→512), ReLU, Dropout(0.5), Linear(512→N))
self.head_gps       = nn.Sequential(Linear(2048+N→512), ReLU, Dropout(0.5), Linear(512→4))
```
- Dropout plus agressif : `0.5`
- GPS conditionné par les probabilités pays (concat features + cls_probs)

**Code collègue**
```python
self.cls_head = nn.Sequential(Linear(2048→512), ReLU, Dropout(0.4), Linear(512→N))
self.reg_head = nn.Sequential(Linear(2048+N→512), ReLU, Dropout(0.4), Linear(512→4))
```
- Dropout plus doux : `0.4`
- Même conditionnement GPS par pays ✅

---

### 4. embed_detach

**Ton code** — `embed_detach=True` par défaut dans tes derniers entraînements :
```python
embed = cls_probs.detach() if self.embed_detach else cls_probs
reg_input = torch.cat([feats, cls_probs], dim=1)  # ← bug : utilise cls_probs au lieu de embed
```
⚠️ **Bug détecté** — tu concatènes `cls_probs` au lieu de `embed` dans `reg_input`. `embed_detach` n'a donc aucun effet dans ton forward.

**Code collègue** — `embed_detach=False` par défaut, implémentation correcte :
```python
embed = cls_probs.detach() if self.embed_detach else cls_probs
reg_input = torch.cat([feats, embed], dim=1)  # ✅ utilise embed
```

---

### 5. Curriculum lambda

**Ton code** — curriculum binaire strict :
```python
if epoch < NUM_EPOCH_PHASE1:
    loss = loss_country        # GPS désactivé
else:
    loss = loss_country + LAMBDA_REG * loss_gps  # GPS activé brutalement
```
Choc brutal au passage phase 1→2 → cause des NaN.

**Code collègue** — GPS actif dès le début, curriculum sur le poids de la classification :
```python
lambda_cls = 10 if epoch < PHASE1 else 1  # classification forte puis faible
loss = lambda_reg * loss_reg + lambda_cls * loss_cls
```
GPS toujours actif — pas de choc, plus stable.

---

### 6. Optimiseur

**Ton code** — lr différentiés par groupe de paramètres :
```python
optimizer = Adam([
    {'params': backbone_params,              'lr': 1e-4},
    {'params': model.head_countries.params, 'lr': 1e-3},
    {'params': model.head_gps.params,       'lr': 1e-3},
])
```
Backbone plus prudent, têtes apprennent plus vite.

**Code collègue** — lr unique pour tous les paramètres :
```python
optimizer = Adam(model.parameters(), lr=1.3e-4, weight_decay=1.9e-4)
```
Plus simple, hyperparamètres issus d'Optuna → `weight_decay=1.9e-4` optimisé.

---

### 7. Scheduler

**Ton code** — pas de scheduler.

**Code collègue** — CosineAnnealingLR :
```python
scheduler = CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)
```
Le lr décroît progressivement jusqu'à 0 — évite les oscillations en fin d'entraînement.

---

### 8. Gel/dégel du backbone

**Ton code** — gel phase 1, dégel phase 2 :
```python
# Phase 1 : backbone gelé
for name, module in model.named_children():
    if name not in ['head_countries', 'head_gps']:
        param.requires_grad = False

# Phase 2 : tout dégelé
for param in model.parameters():
    param.requires_grad = True
```

**Code collègue** — tout dégelé dès le début (`unfreeze_all=True`) :
```python
for param in model.parameters():
    param.requires_grad = True
```
Pas de curriculum de gel — le backbone s'adapte dès le début.

---

### 9. Dataset

**Ton code** — `GeoGuesserIADataset` avec `LabelEncoder` sklearn :
```python
self.le = LabelEncoder()
self.le.fit(self.df['country'])
```
Le LabelEncoder est sauvegardé séparément en `.pkl` — nécessaire pour décoder les prédictions.

**Code collègue** — encodage manuel avec dict :
```python
countries = sorted(data["country"].unique().tolist())
self.country_to_idx = {c: i for i, c in enumerate(countries)}
```
Sauvegardé directement dans le `.pt` — plus autonome, pas besoin de fichier `.pkl` séparé.

---

### 10. Clip gradient

**Ton code** — clipping très agressif :
```python
clip_grad_norm_(model.parameters(), max_norm=0.5)
```

**Code collègue** — clipping plus permissif :
```python
clip_grad_norm_(model.parameters(), max_norm=5.0)
```
`0.5` protège mieux contre les explosions mais peut ralentir l'apprentissage.

---

### Tableau récapitulatif

| Aspect | Ton code | Code collègue |
|---|---|---|
| Backbone | ResNet50 + CBAM | ResNet50 standard |
| Attention | CBAM dans chaque Bottleneck | Aucune |
| Dropout | 0.5 | 0.4 |
| embed_detach | Bug — inactif | Correct |
| Curriculum GPS | Binaire (choc phase 2) | GPS actif dès début |
| lr | Différentiés 1e-4/1e-3 | Unique 1.3e-4 |
| Scheduler | Aucun | CosineAnnealingLR |
| Gel backbone | Phase 1 gelé | Jamais gelé |
| Label encoding | LabelEncoder pkl | Dict dans .pt |
| Grad clipping | 0.5 | 5.0 |
| Weight decay | Non | 1.9e-4 |