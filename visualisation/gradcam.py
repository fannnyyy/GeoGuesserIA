"""
Mise en place :
GradCAM sur layer4 standard    → ce que ResNet regarde sans CBAM
GradCAM sur BottleneckWithCBAM → ce que CBAM force le réseau à regarder
=> comparaison des effets de CBAM sur le réseau
layer4[-1].bottleneck.conv3  →  GradCAM "ResNet pur"
layer4[-1].cbam              →  GradCAM "après attention CBAM"

Étape 1 — Forward pass + hook
Tu enregistres les activations d'une couche cible pendant le forward.
Étape 2 — Backward pass
Tu calcules le gradient de la classe prédite par rapport à ces activations.
Étape 3 — Pondération
Tu multiplies les activations par leurs gradients moyennés, puis tu passes par 
ReLU — ça donne une heatmap de la taille de la feature map.
heatmap = ReLU( Σ gradient_moyen_canal_k * activation_canal_k )
Puis tu upsamples la heatmap à la taille de l'image originale et tu la superposes.


Wrapp du modele pour avoir la bonne sortie demander par gradcam

Rouge vif / jaune  →  attention TRÈS forte  →  "je regarde ici"
Rouge pâle         →  attention faible      →  "je regarde un peu partout"
Bleu               →  attention nulle       →  "j'ignore complètement"

"""

import torch
import torch.nn as nn
import joblib
import sys

sys.path.append("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/cnn_with_attention_module")

from resnet_cbam import GeoGussrAttentionMultiTask, val_test_transform, device, test_dataset_final
from PIL import Image
from PIL import ImageDraw, ImageFont
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt

class ModelWrapperGradCAMClassif(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        pred_countries, _ = self.model(x)
        return pred_countries

def preprocess_img(img_path):
    img = Image.open(img_path)
    img = val_test_transform(img).unsqueeze(0).to(device)
    return img


def generate_gradcam(img_path, target_layer, model):
    img = preprocess_img(img_path)
    gradcam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = gradcam(input_tensor=img)
    return img, grayscale_cam


def compare_gradcam(img_path, model, le):
    rgb_img = np.array(Image.open(img_path).resize((224,224)).convert("RGB")).astype(np.float32) / 255.0
    
    
    with torch.no_grad():
        img_tensor = preprocess_img(img_path)
        pred = model(img_tensor)
        probs = torch.softmax(pred, dim=1)
        pred_class = pred.argmax(dim=1).item()
        confidence = probs[0, pred_class].item() * 100 
        pred_country = le.inverse_transform([pred_class])[0]
    
   
    gradcam_before = generate_gradcam(img_path, model.model.layer4[-1].bottleneck.conv3, model)
    visualization_before = show_cam_on_image(rgb_img, gradcam_before[1][0])
    
    gradcam_after = generate_gradcam(img_path, model.model.layer4[-1].cbam, model)
    visualization_after = show_cam_on_image(rgb_img, gradcam_after[1][0])

    W, H = 224, 224
    BAND = 50  
    BORDER = 3  
    PADDING = 10 
    TITLE_Y = 35
    
    total_w = W * 2 + PADDING + BORDER * 4
    total_h = H + BAND + BORDER * 2 + 40

    final_img = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(final_img)

    text = f"Prediction: {pred_country} ({confidence:.1f}%)"
    bbox = draw.textbbox((0, 0), text)
    text_w = bbox[2] - bbox[0]
    draw.text(((total_w - text_w) // 2, 10), text, fill=(0, 0, 0))

    bbox = draw.textbbox((0, 0), "Before CBAM")
    tw = bbox[2] - bbox[0]
    draw.text((BORDER + (W - tw) // 2, TITLE_Y), "Before CBAM", fill=(0,0,0))

    bbox = draw.textbbox((0, 0), "After CBAM")
    tw = bbox[2] - bbox[0]
    draw.text((BORDER*3 + PADDING + W + (W - tw) // 2, TITLE_Y), "After CBAM", fill=(0,0,0))

    draw.rectangle([BORDER, BAND, BORDER + W, BAND + H], outline=(0,0,0), width=BORDER)
    draw.rectangle([BORDER*3 + PADDING + W, BAND, BORDER*3 + PADDING + W*2, BAND + H], outline=(0,0,0), width=BORDER)


    final_img.paste(Image.fromarray((visualization_before * 255).astype(np.uint8)), (BORDER, BAND))
    final_img.paste(Image.fromarray((visualization_after * 255).astype(np.uint8)), (BORDER*3 + PADDING + W, BAND))

    plt.imsave('gradcam_comparison.png', np.array(final_img))


    


if __name__ == '__main__':
    le = joblib.load("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/cnn_with_attention_module/label_encoder.pkl")
    num_countries = len(le.classes_)
    
    model = GeoGussrAttentionMultiTask(num_countries)
    model.load_state_dict(torch.load("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/cnn_with_attention_module/geoguessr_model_attention_classif_reg.pt", weights_only=True))
    model = ModelWrapperGradCAMClassif(model)
    model.to(device)
    model.eval()
    img_path = test_dataset_final.dataset.image_index[str(test_dataset_final.dataset.df.iloc[test_dataset_final.indices[0]]['id'])]
    compare_gradcam(img_path, model,le)




