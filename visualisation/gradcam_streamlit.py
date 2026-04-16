from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image, ImageDraw
from pytorch_grad_cam.utils.image import show_cam_on_image

def make_gradcam_comparison(pil_img, model_inner, le):   

    class WrapperClassif(nn.Module):
        def __init__(self, m): super().__init__(); self.model = m
        def forward(self, x):
            pred_countries, _ = self.model(x)
            return pred_countries
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform_img(pil_img.convert("RGB")).unsqueeze(0)
    rgb_img = np.array(pil_img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
    wrapper = WrapperClassif(model_inner)
    wrapper.eval()
    
    with torch.no_grad():
        pred = wrapper(img_tensor)
        probs = torch.softmax(pred, dim=1)
        idx = pred.argmax(dim=1).item()
        confidence = float(probs[0, idx]) * 100
        country = le.inverse_transform([idx])[0]
    
    cam_before = GradCAM(model=wrapper, target_layers=[wrapper.model.layer4[-1].bottleneck.conv3])
    hm_before = cam_before(input_tensor=img_tensor)
    viz_before = show_cam_on_image(rgb_img, hm_before[0])
    
    cam_after = GradCAM(model=wrapper, target_layers=[wrapper.model.layer4[-1].cbam])
    hm_after = cam_after(input_tensor=img_tensor)
    viz_after = show_cam_on_image(rgb_img, hm_after[0])
    
    W, H = 224, 224
    BAND, BORDER, PAD = 55, 3, 10
    total_w = W * 2 + PAD + BORDER * 4
    total_h = H + BAND + BORDER * 2
    final = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(final)
    
    title = f"Prédiction : {country} ({confidence:.1f}%)"
    bbox = draw.textbbox((0, 0), title)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 8), title, fill=(30, 30, 30))
    
    for label, x_offset in [("Avant CBAM", BORDER), ("Après CBAM", BORDER * 3 + PAD + W)]:
        bbox = draw.textbbox((0, 0), label)
        lw = bbox[2] - bbox[0]
        draw.text((x_offset + (W - lw) // 2, 30), label, fill=(80, 80, 80))
    
    draw.rectangle([BORDER, BAND, BORDER + W, BAND + H], outline=(30, 30, 30), width=BORDER)
    draw.rectangle([BORDER * 3 + PAD + W, BAND, BORDER * 3 + PAD + W * 2, BAND + H],
                   outline=(30, 30, 30), width=BORDER)
   
    final.paste(Image.fromarray((viz_before * 255).astype(np.uint8)), (BORDER, BAND))
    final.paste(Image.fromarray((viz_after * 255).astype(np.uint8)), (BORDER * 3 + PAD + W, BAND))
    
    return final



def _make_heatmap_image(pil_img, heatmap, title):
    """Assemble une image finale avec bande titre + heatmap."""

    rgb_img = np.array(pil_img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
    viz = show_cam_on_image(rgb_img, heatmap[0])
    
    W, H = 224, 224
    BAND, BORDER = 40, 3
    final = Image.new("RGB", (W + BORDER * 2, H + BAND + BORDER), (245, 245, 245))
    draw = ImageDraw.Draw(final)
    bbox = draw.textbbox((0, 0), title)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 10), title, fill=(30, 30, 30))
    draw.rectangle([BORDER, BAND, BORDER + W, BAND + H], outline=(30, 30, 30), width=BORDER)
    final.paste(Image.fromarray((viz * 255).astype(np.uint8)), (BORDER, BAND))
    return final


def make_gradcam_classif_reg(pil_img, model, idx_to_country):
    """GradCAM pour GeoResNetClassifRegress, cible : logits pays."""

    class WrapperClassifReg(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
        def forward(self, x):
            feats = self.model.backbone(x)
            cls_logits = self.model.cls_head(feats)
            return cls_logits

    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform_img(pil_img.convert("RGB")).unsqueeze(0)
    wrapper = WrapperClassifReg(model)
    wrapper.eval()
    with torch.no_grad():
        logits = wrapper(img_tensor)
        idx = logits.argmax(dim=1).item()
        country = idx_to_country.get(idx, str(idx))
    cam = GradCAM(model=wrapper, target_layers=[wrapper.model.backbone.layer4[-1]])
    heatmap = cam(input_tensor=img_tensor)
    return _make_heatmap_image(pil_img, heatmap, f"GradCAM, Pays prédit : {country}")



def make_gradcam_classif_cells(pil_img, model):
    """GradCAM pour GeoResNetClassif, cible : cellule prédite."""

    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform_img(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_tensor)
        idx = logits.argmax(dim=1).item()
    cam = GradCAM(model=model, target_layers=[model.backbone.layer4[-1]])
    heatmap = cam(input_tensor=img_tensor)
    return _make_heatmap_image(pil_img, heatmap, f"GradCAM, Cellule #{idx}")


def make_gradcam_regression(pil_img, model):
    """GradCAM pour GeoResNet, cible : sin(lat) sortie[:,0]."""
    class SinLatTarget:
        def __call__(self, output):
            if output.dim() == 1:
                return output[0]   
            return output[:, 0] 
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform_img(pil_img.convert("RGB")).unsqueeze(0)
    cam = GradCAM(model=model, target_layers=[model.backbone.layer4[-1]])
    heatmap = cam(input_tensor=img_tensor, targets=[SinLatTarget()])
    return _make_heatmap_image(pil_img, heatmap, "GradCAM, sin(lat)")

