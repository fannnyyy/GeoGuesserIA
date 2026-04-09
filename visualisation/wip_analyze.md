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