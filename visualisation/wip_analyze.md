hypothèse mauvais résultat :
pas assez d'indice visuel dans les images
déséquilibrage


Indice visuel : 
Load road 
0  → Water
1  → Evergreen Needleleaf Forest    ← forêt dense
2  → Evergreen Broadleaf Forest     ← forêt tropicale (dominant!)
3  → Deciduous Needleleaf Forest
4  → Deciduous Broadleaf Forest
5  → Mixed Forest
6  → Closed Shrublands
7  → Open Shrublands
8  → Woody Savannas
9  → Savannas
10 → Grasslands
11 → Permanent Wetlands

Observation immédiate — les classes 1, 2, 4, 5 (forêts) dominent largement 
→ beaucoup d'images avec végétation dense, peu d'indices géographiques discriminants. 
C'est cohérent avec ton hypothèse.

Pour les embeddings t-SNE
Oui tu peux faire les 3 modèles — mais il faut les embeddings intermédiaires, 
pas les prédictions finales. Voici ce qui est disponible :
DINO KNN     → bank_features déjà sauvegardés ✅ direct
ResNet CBAM  → extraire via hook sur layer4[-1] ⚠️ à calculer sur N images
ResNet classif → idem ⚠️
Pour CBAM et ResNet, il faut passer un batch d'images dans le modèle et récupérer les features


Land Cover — barplot MODIS avec % + composition par pays Top 10
Road Index — distribution + road index médian par type de terrain + 3 métriques
t-SNE — bouton pour lancer le calcul, choix du modèle (DINO ou CBAM), colorié 
par pays ou land cover

Le t-SNE DINO utilise directement le bank_features déjà calculé — instantané. 
Le t-SNE CBAM extrait les features en passant des images dans le backbone — plus lent.

## t-SNE — explication simple

---

## C'est quoi t-SNE ?

t-SNE (t-distributed Stochastic Neighbor Embedding) est un 
algorithme de **réduction de dimension** — il prend des vecteurs 
de haute dimension (ex: 2048 features d'un ResNet) et les projette 
en 2D pour qu'on puisse les visualiser.

L'idée clé : **les points proches en haute dimension restent proches 
en 2D**. Donc si deux images ont des features similaires, elles 
apparaîtront proches sur le graphe.

---

## Pourquoi ça dépend du modèle ?

Chaque modèle apprend des **représentations différentes** de la même image :

```
Image de route enneigée en Russie
    ↓
ResNet régression → features orientées vers la position GPS
                    → "cette image ressemble à lat=60, lon=40"

ResNet classif    → features orientées vers le pays
                    → "cette image ressemble à RU"

DINOv2            → features visuelles générales self-supervised
                    → "cette image a de la neige, des conifères, une route grise"

CBAM              → features avec attention spatiale
                    → focus sur les panneaux, texture route, végétation
```

---

## Ce que tu peux conclure

```
t-SNE avec clusters bien séparés par pays
→ le modèle a appris des features géographiquement discriminantes ✅

t-SNE avec points mélangés (comme ton DINO)
→ même DINOv2 ne sépare pas bien les pays
→ les images de différents pays se ressemblent trop visuellement
→ c'est la preuve que le dataset est difficile ⚠️
```

---

## L'intérêt pour ta soutenance

Comparer les t-SNE de tes différents modèles permet de montrer :

- Quel modèle a appris les meilleures représentations géographiques
- Pourquoi les performances sont limitées — pas le modèle qui est 
mauvais, mais les **images elles-mêmes qui manquent d'indices discriminants**
- L'effet de CBAM — est-ce que l'attention améliore la séparation des clusters ?


## Texte récapitulatif — Détails d'implémentation Streamlit

---

### Visualisation des couches internes (`get_activations`)

La fonction `get_activations` enregistre les sorties de chaque 
couche via des `forward hooks` PyTorch. Un hook est attaché à chaque 
module enfant direct du modèle pendant un forward pass, puis retiré
 immédiatement après. Les activations collectées sont des tenseurs 
 de formes variables selon la couche :

Les couches convolutionnelles retournent des tenseurs 4D de shape 
`[batch, canaux, hauteur, largeur]` — ce sont les **feature maps 
spatiales** que l'on peut visualiser sous forme de grille. Par exemple 
`layer4` de ResNet50 produit des feature maps de shape `[1, 2048, 7, 7]`
 — 2048 canaux de 7×7 pixels chacun. La couche `avgpool` produit `[1, 2048, 1, 1]`
  — techniquement 4D mais sans information spatiale utile (1×1 pixel). Les couches 
  fully connected et les têtes de classification/régression produisent des tenseurs 
  2D `[batch, n_sorties]` — non visualisables sous forme de grille.

Pour cette raison, seules les activations avec dimensions spatiales 
strictement supérieures à 1×1 sont affichées sous forme de grille 
(`act.shape[2] > 1 and act.shape[3] > 1`). Les autres affichent leur 
shape avec un message explicatif.

---

### Cas particulier de `GeoResNet` (régression pure)

`GeoResNet` intègre sa tête de régression directement dans 
`backbone.fc` — c'est un `nn.Sequential` qui remplace la tête originale 
de ResNet. Cela signifie que `model.children()` ne retourne que deux modules : 
`backbone` (le ResNet complet avec la tête intégrée) et rien d'autre. 
Hooker les enfants directs de `model` ne donne donc aucune feature map 
intermédiaire — seulement la sortie finale `[1, 4]`.

La solution est d'hooker les enfants de `model.backbone` directement 
via `get_activations_resnet`, ce qui expose les couches internes de ResNet 
(`conv1`, `layer1`, `layer2`, `layer3`, `layer4`, `avgpool`, `fc`).

---

### Cas particulier de `GeoGussrAttentionMultiTask` (CBAM)

Ce modèle copie les couches de ResNet50+CBAM via `setattr` 
— donc `self.conv1`, `self.layer1`... `self.layer4`, `self.avgpool` 
sont des attributs directs du modèle. Hooker `model.children()` 
expose directement toutes les couches internes y compris les blocs CBAM, 
ce qui permet de visualiser l'effet de l'attention sur les feature maps à 
chaque niveau.

---

### GradCAM et la dimension de sortie

GradCAM calcule le gradient de la sortie du modèle par rapport aux 
activations d'une couche cible. Il s'attend à recevoir un **scalaire** 
par image comme signal de gradient. Pour les modèles de classification, 
ce scalaire est naturellement le logit de la classe prédite. Pour les modèles 
de régression pure (`GeoResNet`), il n'y a pas de classe — on cible `sin(lat)` 
(sortie `[:,0]`) comme proxy de la direction latitudinale.

La classe `SinLatTarget` doit gérer deux cas selon comment `pytorch_grad_cam` 
passe l'output : soit sous forme de batch `[B, 4]` (shape 2D), soit sous forme
 d'un seul vecteur `[4]` (shape 1D) quand le batch size est 1. D'où la vérification 
 `if output.dim() == 1: return output[0]`.

---

### Wrapper GradCAM pour les modèles multitâches

Les modèles retournant des tuples `(pred_countries, pred_gps)` ou 
`(pred_gps, cls_logits)` sont incompatibles avec `pytorch_grad_cam` 
qui attend un tenseur simple. Un wrapper `nn.Module` intercepte la 
sortie et ne retourne que la composante d'intérêt — `pred_countries` 
pour visualiser ce que le modèle regarde pour prédire le pays. 
Ce wrapper ne modifie pas les poids ni l'architecture — il change 
uniquement ce qui est exposé à la librairie GradCAM.

---

### Cache disque pour le t-SNE

Le calcul t-SNE sur 1000-5000 points prend 1-3 minutes sur CPU. 
Pour éviter de recalculer à chaque rechargement de la page, 
les résultats sont sauvegardés dans un fichier `.npz` (format NumPy compressé) 
nommé selon le modèle et la perplexité — par exemple `tsne_dino_p30.npz`. 
Au prochain lancement, si le fichier existe, il est chargé directement. 
`@st.cache_data` de Streamlit assure en plus un cache en mémoire pour la 
session courante — le fichier disque prend le relais entre les sessions et 
les redémarrages du job SLURM.