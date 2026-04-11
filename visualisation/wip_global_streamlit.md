lancer job
scontrol show job 178142
dans powershell :
ssh -L 8501:NodeList:8501 dce-login
puis dans navigateur :
http://localhost:8501

chnager de port si besoin (8502:NodeList:8501 dce-login, http://localhost:8502)




Ajout confiance de la prédiction
Ajout carte avec coordonnée GPS en fonction du modèle
Visualiser les masques d'attention et voir littéralement 
ce que ton réseau regarde (cf module d'attention)



GradCAM, Visualisation de l'attention des modèles
GradCAM (Gradient-weighted Class Activation Mapping) génère 
une heatmap indiquant les zones de l'image ayant le plus 
influencé la décision du modèle. Les zones rouges/jaunes 
correspondent à une attention forte, les zones bleues à une attention faible ou nulle.
Selon l'architecture, la cible du GradCAM varie :
ResNet50 + CBAM, Deux heatmaps sont générées en parallèle : 
l'une avant le module d'attention CBAM (layer4[-1].bottleneck.conv3) et 
l'autre après (layer4[-1].cbam). Cette comparaison permet de visualiser 
directement l'effet de CBAM, le module tend à concentrer l'attention 
sur des indices géographiques locaux (panneaux, végétation, texture du sol) en réduisant l'attention sur les zones non discriminantes comme le ciel ou le capot de véhicule.
ResNet50 Classif+Reg, GradCAM ciblant les logits de classification 
pays. La heatmap montre les zones utilisées pour identifier le pays, 
indépendamment de la prédiction GPS.
ResNet50 Classif Cellules, GradCAM ciblant la cellule géographique
 k-means prédite. L'attention reflète les indices utilisés pour 
 discriminer entre zones géographiques définies statistiquement 
 plutôt qu'administrativement.
ResNet50/18 Régression pure, GradCAM ciblant sin(lat), la première 
composante de l'encodage sin/cos de la latitude. Contrairement aux 
modèles de classification, il n'y a pas de classe naturelle à cibler, 
on visualise les gradients par rapport à la composante latitudinale, ce 
qui peut révéler des biais géographiques nord/sud dans les features apprises.
ViT-B/16, Les Vision Transformers n'ont pas de feature maps spatiales 
convolutionnelles. On utilise l'Attention Rollout à la place de GradCAM : 
cette méthode propage les matrices d'attention de tous les blocs Transformer 
pour produire une carte d'attention globale sur les patches de l'image, 
révélant quels patches de 16×16 pixels ont le plus contribué à la prédiction GPS.



Structure générale du code
Le fichier streamlit_app.py est organisé en 5 sections distinctes :
1. Imports et configuration globale
En haut du fichier, après les imports standards (torch, numpy, PIL, folium), 
le chemin vers le dossier CBAM est ajouté au sys.path immédiatement, avant
 tout import dynamique. C'est crucial car cbam.py et resnet_cbam.py ne sont 
 pas dans le même dossier que streamlit_app.py. Sans ce sys.path.insert, 
 l'import échouerait même à l'intérieur des fonctions load_*.
2. Définitions des classes modèles
Plutôt que d'importer depuis les fichiers d'entraînement (qui contiennent du 
code non protégé par if __name__ == '__main__' et crasheraient à l'import), 
les classes sont redéfinies directement dans streamlit_app.py : GeoResNet, 
GeoResNetClassif, GeoResNetClassifRegress, ViTGeo. Seul le modèle CBAM est 
importé dynamiquement via load_cbam() car son fichier source est correctement protégé.
3. Fonctions de chargement avec @st.cache_resource
Chaque modèle a sa propre fonction load_* décorée avec @st.cache_resource. 
Ce décorateur est fondamental, il garantit que le modèle n'est chargé depuis 
le disque qu'une seule fois par session Streamlit, peu importe combien de fois 
l'utilisateur change d'image ou clique. Sans ce cache, chaque prédiction rechargerait 
100-500MB depuis le disque, rendant l'application inutilisable.
Chaque load_* adapte le chargement au format de sauvegarde du modèle, certains .pt 
contiennent directement un state_dict, d'autres un dictionnaire avec model_state_dict, 
config, country_to_idx, centers etc. La détection est faite avec isinstance(state, dict) 
and "model_state_dict" in state.
4. Fonctions de prédiction et preprocessing
Le preprocessing utilise une approche sans NumPy bridge (torch.frombuffer) pour éviter 
les conflits de version entre NumPy 1.x et 2.x, un problème rencontré pendant le développement. 
Chaque predict_* appelle le modèle en torch.no_grad() et retourne un dictionnaire standardisé 
{"country", "confidence", "lat", "lon"}, cette interface commune permet à la page de prédiction 
d'afficher les résultats de façon identique quel que soit le modèle.
5. Interface Streamlit, pages et navigation
La navigation est gérée par un st.radio dans la sidebar qui modifie la variable page. 
Selon sa valeur, un bloc if/elif affiche le contenu correspondant. Ce pattern est plus 
simple que le système multipage natif de Streamlit pour ce cas d'usage.

Subtilités importantes
Gestion du GradCAM multi-modèles
Le GradCAM est externalisé dans gradcam_streamlit.py pour garder streamlit_app.py lisible. 
Quatre fonctions distinctes gèrent quatre cas différents selon l'architecture :

make_gradcam_comparison pour CBAM, génère deux heatmaps en parallèle (before/after CBAM) en 
wrappant le modèle pour n'exposer que pred_countries à la librairie pytorch_grad_cam
make_gradcam_classif_reg pour le modèle classif+reg, wrapper similaire qui extrait cls_logits 
depuis le tuple (GPS, logits)
make_gradcam_classif_cells pour la classification k-means, pas de wrapper nécessaire, le forward 
retourne directement les logits
make_gradcam_regression pour la régression pure, utilise une SinLatTarget custom car il n'y a 
pas de classe naturelle à cibler

Découplage prédiction / GradCAM
La prédiction et le GradCAM sont séparés temporellement dans l'interface, la prédiction 
s'affiche d'abord dans col2, puis GradCAM est généré dans col3 avec son propre st.spinner. 
Si GradCAM échoue (erreur, modèle incompatible), la prédiction reste affichée sans interruption 
grâce aux blocs try/except.
Carte Folium
st_folium est appelé avec returned_objects=[] pour éviter que Streamlit ne re-rende toute 
la page à chaque interaction avec la carte (zoom, clic). Sans ce paramètre, chaque déplacement 
sur la carte déclencherait un recalcul complet.
Visualisation des feature maps
L'expander en bas de page utilise des register_forward_hook sur chaque couche enfant directe du 
modèle. Les activations sont collectées pendant un forward pass no_grad, puis affichées sous 
forme de grille 8×N avec colormap inferno sur fond sombre, cohérent avec le thème dark de 
l'application.
DINO KNN, cas particulier
Contrairement aux autres modèles, DINO KNN n'a pas de forward PyTorch classique. Le pipeline est 
en deux étapes : extraction de features via DINOv2 (HuggingFace ou torch.hub), puis recherche 
dans le bank de features pré-calculées sauvegardé dans le .pt. La contrainte pays (filtrage des 
voisins par pays dominant) est réimplémentée dans predict_dino_knn en reproduisant la logique de 
GeoKNNRegressor.predict_from_features.
CSS personnalisé
Le thème dark avec typographie Syne/DM Mono est injecté via st.markdown(CUSTOM_CSS, 
unsafe_allow_html=True) avant tout contenu. Les composants natifs Streamlit sont stylisés 
via des sélecteurs CSS sur les classes générées ([class*="css"], .stButton > button, 
.stSelectbox). Les cards métriques et badges sont du HTML pur injecté via st.markdown(..., 
unsafe_allow_html=True).




Intégration DINOv2 KNN dans Streamlit, problèmes rencontrés
Problème 1, transformers incompatible avec PyTorch
La librairie transformers d'HuggingFace signalait que PyTorch n'était pas trouvé 
malgré son installation via conda. Le conflit venait d'une version de transformers 
installée via pip dans .local qui ne voyait pas le PyTorch conda. Solution abandonnée 
au profit de torch.hub.
Problème 2, torch.hub incompatible Python 3.9
Le dépôt officiel facebookresearch/dinov2 utilise la syntaxe float | None introduite en 
Python 3.10. L'environnement étant en Python 3.9, l'import crashait avec TypeError: 
unsupported operand type(s) for |: 'type' and 'NoneType'. Solution : utiliser timm qui 
propose une implémentation compatible 3.9 via timm.create_model('vit_large_patch14_dinov2.lvd142m', 
pretrained=True).
Problème 3, Taille d'entrée 518x518
DINOv2 via timm attend par défaut des images de 518×518 pixels (patch size 14, 37×37 patches). 
Les images préprocessées à 224×224 causaient une erreur Input height (224) doesn't match model (518).
 Deux solutions : adapter le transform à 518×518, ou forcer img_size=224 au chargement du modèle 
 via timm.create_model(..., img_size=224), la seconde option est plus cohérente avec les 
 features du bank qui ont été extraites à 224×224 pendant l'entraînement.

Subtilité, extraction des features
Contrairement à AutoModel de HuggingFace qui retourne un objet avec pooler_output et 
last_hidden_state, timm expose forward_features() qui retourne directement le tenseur de 
features de shape [B, seq_len, dim]. Le token CLS (indice 0) est extrait avec features[:, 0] 
pour obtenir la représentation globale de l'image de dimension 1024.

Subtilité, normalisation des features
Les features DINOv2 doivent être normalisées sur la sphère unité avant la recherche KNN, 
la similarité cosinus utilisée par le KNN n'est correcte que si les vecteurs sont normalisés. 
Le bank de features a été sauvegardé en half() (float16) pour économiser la mémoire, il faut 
le convertir en float() avant les calculs.

Subtilité, reconstruction du KNN
Le fichier .pt DINO contient le bank de features pré-calculées mais pas l'objet GeoKNNRegressor 
lui-même. La logique de prédiction (pondération par température, contrainte pays, moyenne pondérée 
des vecteurs cibles) est réimplémentée directement dans predict_dino_knn sans recréer la classe 
complète.


Page 5, Analyse du dataset
├── Section 1 : Distribution land_cover
│   ├── Barplot des classes (avec noms lisibles)
│   └── Performance par land_cover (haversine moyenne)
├── Section 2 : Road index
│   ├── Distribution (histogram)
│   └── Corrélation road_index vs erreur GPS
└── Section 3 : t-SNE embeddings
    ├── Sélection du modèle (DINO / CBAM / ResNet)
    ├── Coloré par pays (top 15)
    └── Coloré par land_cover

