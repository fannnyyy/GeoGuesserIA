## Justification détaillée de chaque modification

---

## Batch 1 — Architecture de base

**`haversine_loss` fonction**

La MSE sur lat/lon bruts est géographiquement incorrecte — elle traite une erreur de 10° en longitude comme identique partout sur Terre. Or 10° près du pôle = ~100km, près de l'équateur = ~1100km. La Haversine calcule la vraie distance en km sur la sphère terrestre en tenant compte de la courbure.

---

**`BottleneckWithCBAM` + `resnet50_cbam`**

Un CNN classique traite toutes les zones de l'image avec la même importance. Pour GeoGuessr les indices géographiques sont petits et localisés — un panneau de rue, une plaque minéralogique, un type de poteau électrique. CBAM force le réseau à apprendre deux choses :

- **Channel attention** → quels détecteurs de features sont utiles pour cette image
- **Spatial attention** → où dans l'image se trouvent les indices géographiques

Le wrapper `BottleneckWithCBAM` permet d'injecter CBAM dans chaque bloc Bottleneck du ResNet50 pretrained **sans réécrire toute l'architecture** — on garde les poids ImageNet intacts via `setattr`.

---

**Deux têtes — classif pays + régression GPS**

Deux tâches complémentaires partagent le même backbone. Le partage permet au réseau d'apprendre des features utiles pour les deux tâches simultanément — les features pays et GPS sont corrélées géographiquement. La classification pays donne un signal fort et facile à apprendre, ce qui aide le backbone à développer des représentations géographiques pertinentes.

---

**`forward` avec `tensor([90, 180])`**

La tête GPS sort des valeurs via Tanh entre -1 et +1. Pour que la Haversine loss reçoive des degrés réels elle doit dénormaliser :
```
tanh → [-1, +1] * [90, 180] → [-90/+90, -180/+180]
```
La dénormalisation est dans le forward et pas dans la loss car la loss doit rester générique et réutilisable.

---

**`GeoGuesserIADataset` avec `LabelEncoder` et `image_index`**

Le pays est une string `"France"` dans le CSV — le réseau a besoin d'un entier. `LabelEncoder` construit un mapping stable `"France"→0`, `"Japan"→1`... dans `__init__` une seule fois, puis applique la transformation dans `__getitem__` à chaque accès. Si on le faisait dans `__getitem__`, il serait recréé pour chaque image — très lent et potentiellement incohérent.

L'`image_index` dict scanne tous les sous-dossiers une seule fois au démarrage — évite de rechercher le bon sous-dossier (`00/`, `01/`...) à chaque accès image ce qui serait très lent sur 50k images.

---

**Train/val/test split**

Sans séparation, le modèle pourrait mémoriser les images plutôt qu'apprendre des features géographiques généralisables. La validation permet de détecter l'overfitting pendant l'entraînement. Le test set ne sert qu'à l'évaluation finale — il n'est jamais vu pendant l'entraînement ni utilisé pour les décisions d'hyperparamètres.

---

**`CrossEntropyLoss` + `Adam` avec lr différents**

`CrossEntropyLoss` intègre le softmax implicitement — plus stable numériquement que d'appliquer softmax puis NLLLoss séparément. Les learning rates différenciés viennent du fait que le backbone est pretrained ImageNet — ses poids sont précieux et fragiles. Un grand lr les écraserait et on perdrait le bénéfice du transfer learning. Les têtes partent de zéro et ont besoin d'un grand lr pour converger rapidement.

---

**2 phases d'entraînement**

Les têtes sont initialisées aléatoirement — si on dégèle tout dès le début, leurs gradients chaotiques remontent dans le backbone et détruisent les features ImageNet. Phase 1 gèle le backbone pour stabiliser les têtes d'abord. Phase 2 dégèle tout pour le fine-tuning complet avec des gradients propres.

---

## Batch 2 — Optimisation de l'entraînement

**`num_workers=4` + `pin_memory=True`**

Sans `num_workers`, le chargement des images se fait dans le thread principal — le GPU attend que le CPU charge chaque batch, créant un goulot d'étranglement. Avec `num_workers=4`, quatre processus préchargent les batchs en parallèle pendant que le GPU entraîne. `pin_memory=True` accélère le transfert CPU→GPU en utilisant la mémoire paginée plutôt que la mémoire paginable. Le run passait de ~10h à ~3-5h.

---

## Batch 3 — Data augmentation et lambda

**`RandomHorizontalFlip` + `ColorJitter` sur train seulement**

Une image Street View retournée horizontalement reste géolocalisable — le réseau ne doit pas apprendre que "les voitures vont à droite = France" car ce serait un biais fragile. L'augmentation double artificiellement la diversité du dataset et réduit l'overfitting. Sur val/test on ne met pas d'augmentation car on veut des résultats reproductibles et comparables à chaque epoch.

---

**Augmentation de `lambda_gps` + normalisation par 20000**

Avec `lambda_gps=0.001` la loss GPS contribuait ~7.8 à la loss totale mais son **gradient** était trop faible pour modifier les poids — la tête GPS ne recevait aucun signal d'apprentissage utile et restait bloquée à ~7800km. La normalisation par 20000 (distance maximale sur Terre) ramène la Haversine entre 0 et 1, comparable à la CrossEntropy, permettant un équilibre entre les deux losses.

---

## Batch 4 — GPS conditionné, sin/cos, curriculum

**`HaversineLoss` classe au lieu de fonction**

Cohérence avec l'écosystème PyTorch — tous les critères (`CrossEntropyLoss`, `MSELoss`...) sont des classes `nn.Module`. Ça permet d'instancier avec des paramètres configurables (`radius=6371`) et d'utiliser la même syntaxe que `criterion_countries` dans la boucle d'entraînement.

---

**Tête GPS : 4 sorties sans Tanh + `F.normalize` par paires**

L'encodage `tanh * [90, 180]` avait deux problèmes fondamentaux.

Premier problème — le **méridien 180°** : deux points à `lon=-179°` et `lon=+179°` sont à 2km l'un de l'autre dans le Pacifique, mais leur représentation tanh est aux deux extrêmes opposés (`-0.994` et `+0.994`), créant une énorme pénalité artificielle pour une prédiction presque correcte.

Deuxième problème — la **saturation de Tanh** : près de ±1, les gradients de Tanh deviennent quasi nuls, ce qui bloque l'apprentissage pour les coordonnées extrêmes (pôles, méridien 180°).

L'encodage sin/cos résout les deux problèmes — les coordonnées vivent sur un cercle unité sans discontinuité. `F.normalize` par paires contraint chaque paire `(sin, cos)` à rester sur son cercle unité respectif, garantissant des valeurs valides pour `atan2` dans la Haversine.

---

**`label_gps` en sin/cos dans le Dataset**

Les labels doivent correspondre exactement au format de sortie du modèle pour que la loss puisse comparer correctement. Puisque la tête GPS sort `[sin_lat, cos_lat, sin_lon, cos_lon]`, les labels doivent être dans le même espace. La conversion `math.radians` puis `sin/cos` se fait une fois par image dans `__getitem__` — pas dans le forward pour garder la séparation données/modèle propre.

---

**GPS conditionné par pays**

```python
reg_input = torch.cat([feats, cls_probs], dim=1)  # 2048 + num_countries
```

C'est la modification architecturale la plus importante. La tête GPS indépendante ne convergait pas car elle devait apprendre à localiser précisément sans aucun contexte géographique — une tâche trop difficile depuis des features génériques seules.

En injectant les probabilités pays comme entrée supplémentaire, la tête GPS dispose d'un **prior géographique** — elle sait dans quelle région du monde chercher avant de prédire les coordonnées précises. C'est une approche **coarse-to-fine** : le pays donne le contexte global (~2000km de précision), le GPS affine localement. Le résultat visible : GPS passait de ~7800km bloqué à ~1500km en convergence.

---

**Curriculum lambda**

```python
if epoch < NUM_EPOCH_PHASE1:
    loss = loss_country           # GPS ignoré
else:
    loss = loss_country + LAMBDA_REG * loss_gps
```

Avec `LAMBDA_CLS_HIGH=10` et la Haversine à ~5000km, même en multipliant pays par 10 la GPS représentait ~99% de la loss totale — les gradients GPS écrasaient complètement le signal pays. La tête pays ne pouvait pas apprendre, et donc les probabilités pays injectées dans la tête GPS étaient mauvaises, créant un cercle vicieux.

En désactivant GPS complètement en phase 1, le pays apprend correctement. En phase 2, le GPS conditionné peut utiliser de bonnes probabilités pays comme contexte pour apprendre efficacement la localisation fine.

## Batch 5 — Découplage des gradients et curriculum corrigé

---

**`embed_detach`**

```python
embed = cls_probs.detach() if self.embed_detach else cls_probs
reg_input = torch.cat([feats, embed], dim=1)
```

**Le problème sans detach :**

Sans `detach()`, le gradient de la loss GPS remonte à travers `cls_probs` jusque dans la tête pays :

```
loss_gps → backward → reg_head → cls_probs → cls_head
                                      ↑
                          gradient GPS perturbe la tête pays
```

La tête pays reçoit donc deux signaux contradictoires — son propre gradient CrossEntropy ET le gradient GPS qui lui dit "arrange tes probabilités pour que le GPS soit meilleur". Ces deux signaux se combattent et destabilisent l'apprentissage du pays.

**Avec `detach()` :**

```
loss_gps → backward → reg_head → cls_probs.detach() → STOP
loss_cls → backward → cls_head  (signal propre uniquement)
```

Les deux têtes apprennent **indépendamment** — la tête pays reçoit uniquement son signal CrossEntropy, la tête GPS utilise les probabilités pays comme contexte sans les modifier. C'est une séparation propre des responsabilités.

---

**Suppression du `lambda_cls` variable + curriculum simplifié**

```python
# Avant — curriculum avec lambda variable
lambda_cls = LAMBDA_CLS_HIGH if epoch < NUM_EPOCH_PHASE1 else LAMBDA_CLS_LOW
loss = lambda_cls * loss_country + LAMBDA_REG * loss_gps

# Après — curriculum binaire
if epoch < NUM_EPOCH_PHASE1:
    loss = loss_country
else:
    loss = loss_country + LAMBDA_REG * loss_gps
```

**Pourquoi supprimer `LAMBDA_CLS_HIGH` et `LAMBDA_CLS_LOW` :**

Le curriculum avec `LAMBDA_CLS_HIGH=10` était insuffisant — même multiplié par 10, la loss pays (~4.0) donnait `10 * 4.0 = 40` contre `1.0 * 5000 = 5000` pour le GPS. Le pays représentait encore moins de 1% de la loss totale et ses gradients étaient noyés.

La solution la plus robuste est **binaire** — soit GPS est complètement absent (phase 1), soit il est présent avec un poids équilibré (phase 2). Pas de compromis qui ne fonctionne pas en pratique. Ça simplifie aussi le code — deux hyperparamètres (`LAMBDA_CLS_HIGH`, `LAMBDA_CLS_LOW`) supprimés, remplacés par une simple condition.

---

## Résumé de l'évolution de l'architecture

```
V1 — GPS indépendant + tanh * [90,180]
     → GPS bloqué à 7800km, pays f1=0.47

V2 — GPS conditionné + sin/cos + curriculum lambda
     → GPS 5900km → 1500km ✅, pays f1=0.015 ❌

V3 — embed_detach + curriculum binaire
     → pays et GPS apprennent indépendamment ✅
     → à tester
```