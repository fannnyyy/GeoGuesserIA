
## Évolution de l'architecture ResNet50 + CBAM

---

## Vue d'ensemble

| Batch | Modification principale | GPS (km) | Pays F1 | Statut |
|-------|------------------------|----------|---------|--------|
| 1 | Architecture de base | ~7800 | 0.40 |  |
| 2 | num_workers + pin_memory | ~7806 | 0.47 |  |
| 3 | Data augmentation + lambda | ~7809 | 0.49 |  GPS bloqué |
| 4 | GPS conditionné + sin/cos + curriculum | ~1519 | 0.015 |  GPS  Pays  |
| 5 | embed_detach + curriculum binaire | NaN ep.7 |, |  explosion |
| 6 | LAMBDA 0.1 + lr 1e-5 + clip 1.0 |, |, |  stabilisation |
| 7 | WeightedRandomSampler + Undersampling US | En cours |, |  |

---

## Batch 1, Architecture de base
*SLURM : 159997, très long*

### Modifications

**`haversine_loss` fonction**
La MSE sur lat/lon bruts est géographiquement incorrecte, elle traite une erreur de 10° en longitude comme identique partout sur Terre. Or 10° près du pôle = ~100km, près de l'équateur = ~1100km. La Haversine calcule la vraie distance en km sur la sphère terrestre en tenant compte de la courbure.

**`BottleneckWithCBAM` + `resnet50_cbam`**
Un CNN classique traite toutes les zones de l'image avec la même importance. Pour GeoGuessr les indices géographiques sont petits et localisés, un panneau de rue, une plaque minéralogique, un type de poteau électrique. CBAM force le réseau à apprendre deux choses :
- **Channel attention** : quels détecteurs de features sont utiles pour cette image
- **Spatial attention** : où dans l'image se trouvent les indices géographiques

Le wrapper `BottleneckWithCBAM` permet d'injecter CBAM dans chaque bloc Bottleneck du ResNet50 pretrained **sans réécrire toute l'architecture**, on garde les poids ImageNet intacts via `setattr`.

**Deux têtes, classif pays + régression GPS**
Deux tâches complémentaires partagent le même backbone. Le partage permet au réseau d'apprendre des features utiles pour les deux tâches simultanément, les features pays et GPS sont corrélées géographiquement. La classification pays donne un signal fort et facile à apprendre, ce qui aide le backbone à développer des représentations géographiques pertinentes.

**`GeoGuesserIADataset` avec `LabelEncoder` et `image_index`**
Le pays est une string `"France"` dans le CSV, le réseau a besoin d'un entier. `LabelEncoder` construit un mapping stable `"France":0`, `"Japan":1`... dans `__init__` une seule fois. L'`image_index` dict scanne tous les sous-dossiers une seule fois au démarrage, évite de rechercher le bon sous-dossier (`00/`, `01/`...) à chaque accès image.

**2 phases d'entraînement**
Les têtes sont initialisées aléatoirement, si on dégèle tout dès le début, leurs gradients chaotiques remontent dans le backbone et détruisent les features ImageNet. Phase 1 gèle le backbone pour stabiliser les têtes d'abord. Phase 2 dégèle tout pour le fine-tuning complet avec des gradients propres.

### Résultats
```
Epoch 1/25, très long (pas de num_workers)
GPS : ~7800 km 
Pays F1 : 0.40
```
> **Diagnostic :** GPS complètement bloqué, confirme que le problème est architectural, pas juste un manque d'epochs. Le pays apprend mais le GPS ne reçoit aucun signal utile.

---

## Batch 2, Optimisation du chargement des données
*SLURM : 160062*

### Modifications

**`num_workers=4` + `pin_memory=True`**
Sans `num_workers`, le chargement des images se fait dans le thread principal, le GPU attend que le CPU charge chaque batch, créant un goulot d'étranglement. Avec `num_workers=4`, quatre processus préchargent les batchs en parallèle pendant que le GPU entraîne. `pin_memory=True` accélère le transfert CPU:GPU en utilisant la mémoire paginée. Le run passait de ~10h à ~3-5h.

### Résultats
```
Epoch 25/25
train      | loss_total: 8.5414 | loss_country: 0.7350 | loss_gps: 7806.5km | f1: 0.4002
validation | loss_total: 9.0822 | loss_country: 1.2917 | loss_gps: 7790.5km | f1: 0.4743
```
> **Diagnostic :** GPS bloqué à 7806km , Pays F1 0.47 . Le pays apprend bien mais GPS complètement mort. Confirme que le problème est architectural, pas juste un manque d'epochs.

---

## Batch 3, Data augmentation et lambda GPS
*SLURM : 160426*

### Modifications

**`RandomHorizontalFlip` + `ColorJitter` sur train seulement**
Une image Street View retournée horizontalement reste géolocalisable, le réseau ne doit pas apprendre que "les voitures vont à droite = France" car ce serait un biais fragile. L'augmentation double artificiellement la diversité du dataset et réduit l'overfitting. Sur val/test on ne met pas d'augmentation car on veut des résultats reproductibles et comparables.

**Augmentation de `lambda_gps` + normalisation par 20000**
Avec `lambda_gps=0.001` la loss GPS contribuait ~7.8 à la loss totale mais son gradient était trop faible pour modifier les poids, la tête GPS ne recevait aucun signal d'apprentissage utile. La normalisation par 20000 (distance maximale sur Terre) ramène la Haversine entre 0 et 1, comparable à la CrossEntropy.

### Résultats
```
Epoch 25/25
train      | loss_total: 1.1416 | loss_country: 0.7511 | loss_gps: 7809.0km | f1: 0.4289
validation | loss_total: 1.5433 | loss_country: 1.1544 | loss_gps: 7776.9km | f1: 0.4943
```
> **Diagnostic :** GPS toujours bloqué à 7809km , Pays F1 0.49  légère amélioration. Preuve définitive que lambda et normalisation ne suffisent pas, il faut changer l'architecture GPS.

---

## Batch 4, GPS conditionné, encodage sin/cos, curriculum lambda
*Architecture V2*

### Modifications

**Tête GPS : 4 sorties + `F.normalize` par paires**
L'encodage `tanh * [90, 180]` avait deux problèmes fondamentaux :

- **Méridien 180°** : deux points à `lon=-179°` et `lon=+179°` sont à 2km l'un de l'autre dans le Pacifique, mais leur représentation tanh est aux deux extrêmes opposés (`-0.994` et `+0.994`), créant une énorme pénalité artificielle.
- **Saturation de Tanh** : près de ±1, les gradients deviennent quasi nuls, bloquant l'apprentissage pour les coordonnées extrêmes.

L'encodage sin/cos résout les deux problèmes, les coordonnées vivent sur un cercle unité sans discontinuité.

**GPS conditionné par pays**
```python
reg_input = torch.cat([feats, cls_probs], dim=1)  # 2048 + num_countries
```
La tête GPS indépendante ne convergait pas car elle devait localiser précisément sans contexte géographique. En injectant les probabilités pays, la tête GPS dispose d'un **prior géographique**, approche coarse-to-fine : le pays donne le contexte global (~2000km), le GPS affine localement.

**Curriculum lambda**
```python
if epoch < NUM_EPOCH_PHASE1:
    loss = loss_country      # GPS ignoré en phase 1
else:
    loss = loss_country + LAMBDA_REG * loss_gps
```

### Résultats
```
Epoch 25/25
train      | loss_total: 1522.9128 | loss_country: 3.3538 | loss_gps: 1519.6km | f1: 0.0153
validation | loss_total: 1800.4628 | loss_country: 3.0325 | loss_gps: 1797.4km | f1: 0.0155
```
> **Diagnostic :** GPS 7800 : 1519km  (-80%). Pays F1 0.015  (effondrement depuis 0.49). Le GPS conditionné fonctionne mais le curriculum lambda a tué le pays, cercle vicieux : mauvaises probas pays : mauvais contexte GPS : gradients GPS perturbent pays.

---

## Batch 5, Découplage des gradients (embed_detach) + curriculum binaire
*Architecture V3*

### Modifications

**`embed_detach=True`**
```python
embed = cls_probs.detach() if self.embed_detach else cls_probs
reg_input = torch.cat([feats, embed], dim=1)
```
Sans `detach()`, le gradient GPS remontait à travers `cls_probs` jusque dans la tête pays, deux signaux contradictoires (CrossEntropy + GPS) se combattaient et déstabilisaient le pays. Avec `detach()`, les deux têtes apprennent indépendamment.

**Curriculum binaire + suppression LAMBDA_CLS**
Même avec `LAMBDA_CLS_HIGH=10`, la loss pays (~4.0) donnait `40` contre `5000` pour le GPS, le pays représentait moins de 1% de la loss totale. Solution binaire : GPS complètement absent en phase 1, présent avec poids équilibré en phase 2.

### Résultats
```
Epoch 5  : loss_country: 3.32   phase 1 normale
Epoch 6  : loss_total: 4385km  ← GPS explose au passage phase 2
Epoch 7  : NaN partout  gradient explosion
```
> **Diagnostic :** `LAMBDA_REG=1.0` trop grand + lr trop élevé en phase 2 : gradients GPS explosent : NaN irréversible.

---

## Batch 6, Stabilisation du passage phase 1:2

### Modifications
- `LAMBDA_REG : 1.0 : 0.1`, réduire l'impact GPS en phase 2
- `lr phase 2 : 1e-4 : 1e-5`, pas plus petits pour éviter l'explosion
- `clip_grad_norm_ : max_norm=5.0 : 1.0`, clipping plus agressif
- Détection NaN, ignorer les batchs corrompus :

```python
if torch.isnan(loss):
    optimizer.zero_grad()
    continue
```

> **Objectif :** passage phase 1:2 stable sans explosion de gradient.

---

## Batch 7, Rééquilibrage du dataset + expérimentations

### Contexte
L'EDA du dataset OSV5M révèle un déséquilibre géographique majeur :
- **US : 869 424 images (~24% du dataset)**, largement dominant
- DE : 189 827 | FR : 169 684 | RU : 169 499
- Pays rares (PK, IN, ID...) : < 50 000 images chacun

**Remarque importante sur le dataset :** Les images OSV5M sont principalement des images de routes (Street View). Il n'y a pas toujours d'indices visuels géographiques forts, les routes se ressemblent entre pays, ce qui rend la tâche de géolocalisation fondamentalement difficile et explique en partie les performances limitées.

**Sous-dossiers supprimés :** Les plus gros sous-dossiers d'OSV5M supprimés contenaient principalement des images US, Europe de l'Ouest et Japon. Cela a paradoxalement pu aider l'équilibre géographique mais a appauvri la diversité visuelle pour ces pays.

### Expériences lancées

| Job SLURM | Modèle | Dataset | Sampler/Équilibrage | Statut | Résultat |
|-----------|--------|---------|---------------------|--------|----------|
| 168742 | ResNet50 classif k-means (modèle M) | rest (500k samples) | WeightedRandomSampler par cellule |  Terminé | À compléter |
| 168758 | ResNet50 + CBAM attention | rest | WeightedRandomSampler par pays |  CANCELED | Timeout DCE |
| 168750 | ResNet50 sans attention (comparaison) | samples | Sans sampler |  FAILED | NaN values |
| 168736 | ResNet50 + CBAM attention | samples | WeightedRandomSampler |  CANCELED + NaN | Timeout + NaN |
| 177594 | ResNet50 + CBAM attention | rest | Undersampling US (cap 30k) |  En cours |, |

### Justification des approches

**WeightedRandomSampler par cellule k-means (modèle M)**
Aligné avec la CrossEntropy géographique, chaque cellule reçoit le même poids d'apprentissage indépendamment de sa fréquence dans le dataset.

**WeightedRandomSampler par pays (modèles attention)**
Cohérent avec la tête de classification pays, les pays rares reçoivent un poids inversement proportionnel à leur fréquence.

**Undersampling US (cap 30k)**
Alternative plus simple et stable au WeightedRandomSampler, supprime directement les images US excédentaires avant l'entraînement. Pas de modification de la logique du DataLoader, compatible avec `shuffle=True`. `LAMBDA_REG` abaissé à `0.001` et `clip_grad_norm` à `0.5` pour éviter les NaN.

### Cause des NaN (jobs 168750 et 168736)
`LAMBDA_REG=0.1` trop grand combiné au dégelage du backbone en phase 2, les gradients GPS explosent et propagent des NaN irréversibles dans tout le réseau.

---

WeightedRandomSampler  : sur-échantillonne les pays rares à chaque batch
                       : tous les pays voient autant de batchs
                       : mais les images US apparaissent moins souvent

Undersampling US       : supprime physiquement des images US du dataset
                       : plus simple, plus transparent
                       : irréversible (images supprimées de l'entraînement)

Tu peux noter que les deux ont été testés, le WeightedRandomSampler n'a pas eu l'effet escompté à cause du bug shuffle=sampler, et l'undersampling US a été préféré pour sa simplicité et sa transparence.
Ce bug signifie que le WeightedRandomSampler n'a jamais fonctionné, les batchs étaient dans l'ordre du CSV sans mélange ni rééquilibrage. On l'a supprimé quand on a corrigé le bug, et remplacé par shuffle=True simple.

## Batch 8, Stabilisation du NaN

Diagnostic
Les trois modèles entraînés (CBAM samples, CBAM sans attention comparaison, 
*CBAM rest undersampling US) ont tous présenté un NaN systématique entre 
l'epoch 12 et 17, précisément au moment du passage de la phase 1 à la phase 2. 
Ce phénomène est identique dans les trois cas, ce qui confirme qu'il ne s'agit pas 
d'un problème de données mais d'un problème architectural lié au curriculum.
La cause : en phase 1, seule la loss_country est optimisée. La tête GPS n'apprend 
pas. Quand la phase 2 démarre, la loss_gps est brutalement ajoutée avec une valeur 
initiale de ~2000-3000 km. Même avec LAMBDA_REG=0.01, cela donne 
loss_total = loss_country + 0.01 × 2500 = 2.2 + 25 = 27. Ce choc de gradient dépasse 
le clip_grad_norm=0.5 et corrompt les poids.

## NaN dans l'entraînement, Diagnostic et solution

---

### Pourquoi les NaN apparaissent

Les NaN sont apparus de manière systématique entre l'epoch 12 et 17 dans tous les modèles testés, précisément au moment du passage de la phase 1 à la phase 2. Ce comportement identique sur trois modèles distincts confirme que le problème est architectural et non lié aux données.

En phase 1, seule la `loss_country` est optimisée, la tête GPS ne reçoit aucun signal de gradient et ses poids restent dans leur état d'initialisation aléatoire. Quand la phase 2 démarre brutalement à l'epoch 10, la `loss_gps` est ajoutée d'un coup avec une valeur initiale de l'ordre de 2000 à 3000 km. Même avec `LAMBDA_REG=0.01`, cela donne une `loss_total` de l'ordre de 25 dès le premier batch de la phase 2, soit un ordre de grandeur supérieur à la `loss_country` qui était autour de 2.

Ce choc de gradient dépasse la capacité du `clip_grad_norm=0.5` à le contenir. Les gradients explosent, les poids prennent des valeurs infinies, et dès qu'une valeur infinie entre dans un calcul, elle produit un NaN. Une fois un NaN introduit dans les poids, il se propage à tous les calculs suivants de façon irréversible, c'est pourquoi tous les epochs suivants restent à NaN même avec la détection et le skip des batchs NaN.

---

### Pourquoi le bug embed_detach aggravait la situation

Un second problème a été identifié en comparant l'implémentation avec le code d'une collègue. Dans le `forward` du modèle, `embed_detach=True` était censé découpler les gradients GPS et pays, empêcher la tête GPS d'envoyer des gradients vers la tête pays via l'embedding. Mais le code utilisait `cls_probs` directement au lieu de `embed` dans la concaténation :

```python
embed = cls_probs.detach() if self.embed_detach else cls_probs
reg_input = torch.cat([feats, cls_probs], dim=1)  #  embed_detach sans effet
```

Ce bug signifie que depuis le début du projet, `embed_detach` n'avait aucun effet, les gradients GPS remontaient toujours dans la tête pays, créant des interférences supplémentaires au moment du passage en phase 2.

---

## Solution, Batch 9

Deux corrections ont été apportées simultanément.

**Correction du bug embed_detach**, utiliser `embed` au lieu de `cls_probs` dans la concaténation :

```python
embed = cls_probs.detach() if self.embed_detach else cls_probs
reg_input = torch.cat([feats, embed], dim=1)  # 
```

**Warmup progressif de lambda_gps**, au lieu d'activer le GPS brutalement à l'epoch 10, le poids de la loss GPS monte progressivement sur 5 epochs. En phase 1, le GPS est présent avec un poids quasi nul (`1e-5`) pour que la tête GPS ne parte pas de zéro. En phase 2, lambda monte de 0 à `LAMBDA_REG` sur 5 epochs :

```python
if epoch < NUM_EPOCH_PHASE1:
    loss = loss_country + 0.00001 * loss_gps    # GPS quasi nul mais présent

else:
    warmup = min((epoch - NUM_EPOCH_PHASE1) / 5.0, 1.0)
    loss = loss_country + (LAMBDA_REG * warmup) * loss_gps
```

L'idée est d'éviter le choc brutal qui causait les NaN. En gardant le GPS légèrement actif dès la phase 1, la tête GPS ne part pas de zéro quand la phase 2 démarre, ses poids ont déjà convergé vers des valeurs raisonnables, et l'augmentation progressive de lambda évite toute explosion de gradient.

`LAMBDA_REG` a également été réduit à `0.0001` pour que la contribution GPS en régime permanent reste comparable à la `loss_country` :

```
loss_gps × 0.0001 = 2000 × 0.0001 = 0.2
loss_country      ≈ 2.0
: ratio GPS/pays  ≈ 10%   équilibré
```

+ détection Nan 

## TODO, À faire

- [ ] **Grad-CAM sur tous les modèles**, visualiser les zones d'attention pour comprendre ce que le réseau regarde (panneaux, végétation, routes, ciel...)
- [ ] **Comparer toutes les combinaisons de modèles** :
  - Avec/sans CBAM
  - Avec/sans WeightedRandomSampler
  - Dataset samples vs rest
  - Modèle M (k-means) vs modèle attention (pays + GPS)
- [ ] **Récupérer les métriques finales** du job 168742 (modèle M 500k samples)
- [ ] **Analyser l'impact** des sous-dossiers supprimés sur les performances par pays
- [ ] **Mentionner dans le rapport** : dataset majoritairement composé d'images de routes sans indices visuels forts : limite intrinsèque des performances

---

## Évolution de l'architecture, Résumé visuel

```
V1, GPS indépendant + tanh * [90,180]
     SLURM 159997 / 160062
     : GPS bloqué à 7800km   |  Pays F1 = 0.47 

        ↓ Batch 3 : augmentation + lambda

     SLURM 160426
     : GPS toujours 7809km   |  Pays F1 = 0.49 
     : Preuve : problème architectural, pas de données

        ↓ Batch 4 : GPS conditionné + sin/cos

V2, GPS conditionné par pays + encodage sin/cos + curriculum
     : GPS 7800 : 1519km  (-80%)  |  Pays F1 = 0.015 
     : Cercle vicieux : gradients GPS perturbent pays

        ↓ Batch 5 : embed_detach + curriculum binaire

V3, Découplage gradients + curriculum binaire
     : Epoch 5 stable   :  Epoch 6-7 : NaN 
     : LAMBDA_REG=1.0 trop grand

        ↓ Batch 6 : LAMBDA 0.1 + lr 1e-5 + clip 1.0

V4, Stabilisation passage phase 2
     : En cours d'évaluation

        ↓ Batch 7 : rééquilibrage dataset

V5, Undersampling US + LAMBDA 0.001 + clip 0.5
     SLURM 177594
     : En cours
```