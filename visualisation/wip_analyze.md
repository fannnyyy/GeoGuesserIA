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