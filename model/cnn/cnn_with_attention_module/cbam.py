"""
Un CNN classique traite tous les pixels et tous les canaux avec la même 
importance. L'attention, c'est apprendre automatiquement à dire "cette 
zone / ce canal compte plus que les autres".

Feature map : [Batch, Canaux C, Hauteur H, Largeur W]
                         ↑            ↑
                   Channel att.   Spatial att.


Image Street View
      ↓
   Backbone CNN (ResNet50/101)
   + modules d'attention intercalés
      ↓
   Feature map globale
      ↙          ↘
Tête Pays       Tête GPS
(classif        (classif de cellules
~200 classes)    géographiques fines
                 ~3000-10000 cellules)

CBAM > AW-Conv + SE seul
Pourquoi ? CBAM = channel attention + spatial attention en séquence. 
La dimension spatiale est critique pour toi : le réseau doit apprendre 
à ignorer le ciel générique et focaliser sur le panneau de rue en bas 
à gauche, ou la texture du trottoir.

AW-Conv est une amélioration fine qui s'ajoute par-dessus, mais c'est 
un gain marginal (+0.5% sur ImageNet). Pour GeoGuessr, le gain de CBAM 
sur la dimension spatiale sera bien plus impactant.

Les indices géographiques dans Street View sont très locaux et petits 
(plaque minéralogique, sens de conduite, type de poteau électrique). 
Envisage de garder des feature maps à haute résolution en entrée du 
module d'attention, ne pas trop downsampler tôt dans le backbone est 
aussi important que le choix du module.


CBAM :
CBAM integrates two sequential attention mechanisms: channel attention 
and spatial attention.

Channel Attention Module: Focuses on “what” is meaningful given an input 
image by emphasizing important channels.
Channel Attention — "Quels canaux regarder ?"
Chaque canal d'une feature map détecte un type de pattern (bords, textures, couleurs...). 
L'idée : certains canaux sont plus utiles que d'autres selon l'image en entrée

Feature map [C, H, W]
      ↓
Global Average Pooling  →  vecteur [C]   (résume chaque canal en 1 scalaire)
      ↓
MLP (FC → ReLU → FC)    →  vecteur [C]   (apprend les inter-dépendances)
      ↓
Sigmoid                 →  vecteur [C] ∈ [0,1]   (poids par canal)
      ↓
Multiplication          →  feature map recalibrée [C, H, W]

-> principe de SE-Net

Spatial Attention Module: Focuses on “where” is the important part of an 
input image by emphasizing important spatial locations.
Feature map [C, H, W]
      ↓
AvgPool + MaxPool sur les canaux  →  2 cartes [1, H, W]
      ↓
Concaténation                     →  [2, H, W]
      ↓
Conv 7×7                          →  [1, H, W]
      ↓
Sigmoid                           →  masque spatial [H, W] ∈ [0,1]
      ↓
Multiplication                    →  feature map recalibrée [C, H, W]

SE et CBAM :
Sortie = Conv(X, W) * Attention(X)

AW-Conv :
Sortie = Conv(X, W * Attention(X))

Recalibrer les activations après coup (SE et CBAM) est une approximation,
on modifie le résultat mais pas la façon dont le réseau "cherche" l'information. 
AW-Conv modifie la recherche elle-même.

CBAM :
Feature map X
      ↓
Channel Attention(X)  →  X' = X * Mc    (recalibre les canaux)
      ↓
Spatial Attention(X') →  X'' = X' * Ms  (recalibre les positions)
      ↓
X'' = feature map finale affinée

https://github.com/Peachypie98/CBAM
"""


import torch 
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = torch.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x

