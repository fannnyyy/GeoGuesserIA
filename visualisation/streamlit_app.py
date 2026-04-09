
import sys
import os
import math
import pickle
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image, ImageDraw
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
from gradcam_streamlit import make_gradcam_comparison, make_gradcam_classif_reg, make_gradcam_classif_cells, make_gradcam_regression
from page_analyse_streamlit import render_analyse_page

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CBAM_SRC = os.path.join(BASE_DIR, "../model/cnn/cnn_with_attention_module")
if CBAM_SRC not in sys.path:
    sys.path.insert(0, CBAM_SRC)

 
PATHS = {
    # ResNet50 + CBAM (ton modèle)
    "cbam_v1": {
        "pt":  os.path.join(BASE_DIR, "../model/saved/geoguessr_model_attention_classif_reg.pt"),
        "pkl": os.path.join(BASE_DIR, "../model/saved/label_encoder.pkl"),
        "src": os.path.join(BASE_DIR, "../model/saved"),
    },
    "cbam_v2": {
        "pt":  os.path.join(BASE_DIR, "../model/saved/geoguessr_model_attention_classif_reg_more_epoch.pt"),
        "pkl": os.path.join(BASE_DIR, "../model/saved/label_encoder_more_epoch.pkl"),
        "src": os.path.join(BASE_DIR, "../model/saved"),
    },
    # ResNet50 classif + reg (collègue)
    "classif_reg": {
        "pt":  os.path.join(BASE_DIR, "../model/saved/resnet50_classif_regress.pt"),
        "pkl": None,
    },
    # ResNet50 classif cellules k-means
    "classif_cells": {
        "pt":   os.path.join(BASE_DIR, "../model/saved/resnet50_classification_210k.pt"),
        "pkl":  None,
        "cells": os.path.join("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/resnet/resnet_full_classif/cells_kmeans.pkl"),
    },
    # ResNet50 geo final (régression pure)
    "resnet50_geo": {
        "pt": os.path.join(BASE_DIR, "../model/saved/resnet50_geo_final.pt"),
    },
    # ResNet18 geo
    "resnet18_geo": {
        "pt": os.path.join(BASE_DIR, "../model/saved/resnet18_geo.pt"),
    },
    # ViT
    "vit": {
        "pt": os.path.join(BASE_DIR, "../model/saved/vit_geo_final_state_dict.pt"),
    },
    # DINO KNN
    "dino_knn": {
        "pt":  os.path.join(BASE_DIR, "../model/saved/dinov2_knn_geo_index.pt"),
        "src": os.path.join(BASE_DIR, "../model/saved"),
    },
}

def add_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


class GeoResNet(nn.Module):
    def __init__(self, backbone="resnet50", hidden_dim=512, n_layers=1, dropout_p=0.4):
        super().__init__()
        if backbone == "resnet50":
            self.backbone = tv_models.resnet50(weights=None)
        else:
            self.backbone = tv_models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        if n_layers == 1:
            head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, 4),
            )
        else:
            head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(), nn.Dropout(dropout_p),
                nn.Linear(hidden_dim // 2, 4),
            )
        self.backbone.fc = head
 
    def forward(self, x):
        x = self.backbone(x)
        lat = F.normalize(x[:, 0:2], dim=1)
        lon = F.normalize(x[:, 2:4], dim=1)
        return torch.cat([lat, lon], dim=1)
 

class GeoResNetClassif(nn.Module):
    def __init__(self, n_cells, hidden_dim=512, dropout_p=0.4):
        super().__init__()
        backbone = tv_models.resnet50(weights=None)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_cells),
        )
 
    def forward(self, x):
        return self.head(self.backbone(x))
 
 
class GeoResNetClassifRegress(nn.Module):
    def __init__(self, num_countries, hidden_dim=512, dropout_p=0.4, embed_detach=False):
        super().__init__()
        self.embed_detach = embed_detach
        backbone = tv_models.resnet50(weights=None)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.cls_head = nn.Sequential(
            nn.Linear(in_feats, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_countries),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(in_feats + num_countries, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 4),
        )
 
    def forward(self, x):
        feats = self.backbone(x)
        cls_logits = self.cls_head(feats)
        cls_probs = torch.softmax(cls_logits, dim=1)
        embed = cls_probs.detach() if self.embed_detach else cls_probs
        raw_coords = self.reg_head(torch.cat([feats, embed], dim=1))
        lat = F.normalize(raw_coords[:, 0:2], dim=1)
        lon = F.normalize(raw_coords[:, 2:4], dim=1)
        return torch.cat([lat, lon], dim=1), cls_logits
 

class ViTGeo(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = tv_models.vit_b_16(weights=None)
        self.net.heads.head = nn.Linear(self.net.heads.head.in_features, 3)
 
    def forward(self, x):
        return self.net(x)
 

def normalize_vec(v):
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)
 
def unitvec_to_latlon(v):
    v = normalize_vec(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    lat = torch.asin(z.clamp(-1, 1))
    lon = torch.atan2(y, x)
    return float(torch.rad2deg(lat).item()), float(torch.rad2deg(lon).item())
 
def sincos_to_latlon(pred):
    lat = math.degrees(math.atan2(float(pred[0, 0]), float(pred[0, 1])))
    lon = math.degrees(math.atan2(float(pred[0, 2]), float(pred[0, 3])))
    return lat, lon


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
def preprocess(pil_img):
    raw = bytearray(pil_img.convert("RGB").resize((224, 224), Image.LANCZOS).tobytes())
    tensor = torch.frombuffer(raw, dtype=torch.uint8).clone()
    tensor = tensor.reshape(224, 224, 3).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    tensor = (tensor - mean) / std
    return tensor.permute(2, 0, 1).unsqueeze(0)


@st.cache_resource
def load_cbam(version="v1"):
    key = f"cbam_{version}"
    cfg = PATHS[key]
    
    from cbam import CBAM
    from resnet_cbam import GeoGussrAttentionMultiTask
    le = joblib.load(cfg["pkl"])
    num_countries = len(le.classes_)
    model = GeoGussrAttentionMultiTask(num_countries)
    state = torch.load(cfg["pt"], map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, le
 
 
@st.cache_resource
def load_classif_reg():
    cfg = PATHS["classif_reg"]
    ckpt = torch.load(cfg["pt"], map_location="cpu", weights_only=False)
    country_to_idx = ckpt["country_to_idx"]
    idx_to_country = {v: k for k, v in country_to_idx.items()}
    num_countries = len(country_to_idx)
    cfg_model = ckpt.get("config", {"hidden_dim": 512, "dropout_p": 0.4})
    model = GeoResNetClassifRegress(
        num_countries=num_countries,
        hidden_dim=cfg_model.get("hidden_dim", 512),
        dropout_p=cfg_model.get("dropout_p", 0.4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, idx_to_country
 
 
@st.cache_resource
def load_classif_cells():
    cfg = PATHS["classif_cells"]
    with open(cfg["cells"], "rb") as f:
        cells = pickle.load(f)
    n_cells = cells["n_cells"]
    centers = cells["centers"]
    ckpt = torch.load(cfg["pt"], map_location="cpu", weights_only=False)
    cfg_model = ckpt.get("config", {"hidden_dim": 512, "dropout_p": 0.4})
    model = GeoResNetClassif(
        n_cells=n_cells,
        hidden_dim=cfg_model.get("hidden_dim", 512),
        dropout_p=cfg_model.get("dropout_p", 0.4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, centers
 
 
@st.cache_resource
def load_geo_resnet(variant="resnet50"):
    key = "resnet50_geo" if variant == "resnet50" else "resnet18_geo"
    cfg = PATHS[key]
    ckpt = torch.load(cfg["pt"], map_location="cpu", weights_only=False)
    p = ckpt.get("hyperparams", {"backbone": variant, "hidden_dim": 512, "n_layers": 1, "dropout_p": 0.4})
    model = GeoResNet(
        backbone=p.get("backbone", variant),
        hidden_dim=p.get("hidden_dim", 512),
        n_layers=p.get("n_layers", 1),
        dropout_p=p.get("dropout_p", 0.4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
 
 
@st.cache_resource
def load_vit():
    cfg = PATHS["vit"]
    model = ViTGeo()
    ckpt = torch.load(cfg["pt"], map_location="cpu", weights_only=False)
    
    # C'est directement un state_dict — pas de dict wrapper
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "net.class_token" in ckpt:
        state = ckpt
    else:
        state = ckpt
    
    model.load_state_dict(state)
    model.eval()
    return model
 
 
@st.cache_resource
def load_dino_knn():
    cfg = PATHS["dino_knn"]
    ckpt = torch.load(cfg["pt"], map_location="cpu", weights_only=False)
    
    # Charger DINOv2 via torch.hub sans transformers
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dino_model.eval()
    
    return dino_model, ckpt
    
def predict_cbam(model, le, tensor):
    with torch.no_grad():
        pred_countries, pred_gps = model(tensor)
    probs = torch.softmax(pred_countries, dim=1)
    idx = probs.argmax(dim=1).item()
    country = le.inverse_transform([idx])[0]
    confidence = float(probs[0, idx]) * 100
    lat, lon = sincos_to_latlon(pred_gps)
    return {"country": country, "confidence": confidence, "lat": lat, "lon": lon}
 
 
def predict_classif_reg(model, idx_to_country, tensor):
    with torch.no_grad():
        pred_gps, cls_logits = model(tensor)
    probs = torch.softmax(cls_logits, dim=1)
    idx = probs.argmax(dim=1).item()
    country = idx_to_country.get(idx, str(idx))
    confidence = float(probs[0, idx]) * 100
    lat, lon = sincos_to_latlon(pred_gps)
    return {"country": country, "confidence": confidence, "lat": lat, "lon": lon}
 
 
def predict_classif_cells(model, centers, tensor):
    with torch.no_grad():
        logits = model(tensor)
    idx = logits.argmax(dim=1).item()
    lat, lon = float(centers[idx][0]), float(centers[idx][1])
    confidence = float(torch.softmax(logits, dim=1)[0, idx]) * 100
    return {"country": None, "confidence": confidence, "lat": lat, "lon": lon,
            "extra": f"Cellule #{idx}"}
 
 
def predict_geo_resnet(model, tensor):
    with torch.no_grad():
        pred = model(tensor)
    lat, lon = sincos_to_latlon(pred)
    return {"country": None, "confidence": None, "lat": lat, "lon": lon}
 
 
def predict_vit(model, tensor):
    with torch.no_grad():
        pred_vec = model(tensor)
    pred_vec = normalize_vec(pred_vec)
    lat, lon = unitvec_to_latlon(pred_vec)
    return {"country": None, "confidence": None, "lat": lat, "lon": lon}




def make_map(lat, lon, country=None):
    m = folium.Map(location=[lat, lon], zoom_start=5, tiles="CartoDB positron")
    popup_text = f"Prédit : {country}<br>Lat: {lat:.4f}° / Lon: {lon:.4f}°" if country else f"Lat: {lat:.4f}° / Lon: {lon:.4f}°"
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(popup_text, max_width=200),
        tooltip="Localisation prédite",
        icon=folium.Icon(color="red", icon="map-marker"),
    ).add_to(m)
    folium.Circle([lat, lon], radius=50000, color="#e74c3c", fill=True, fill_opacity=0.1).add_to(m)
    return m




def get_activations(model, tensor):
    activations, hooks = [], []
    def make_hook():
        def fn(m, inp, out): activations.append(out.detach().cpu())
        return fn
    for layer in model.children():
        hooks.append(layer.register_forward_hook(make_hook()))
    with torch.no_grad():
        model(tensor)
    for h in hooks: h.remove()
    return activations

 
import timm

@st.cache_resource
def load_dino_knn():
    cfg = PATHS["dino_knn"]
    ckpt = torch.load(cfg["pt"], map_location="cpu", weights_only=False)
    
    # DINOv2 via timm — compatible Python 3.9
    dino_model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True)
    dino_model.eval()
    
    return dino_model, ckpt


def extract_dino_features(dino_model, tensor):
    with torch.no_grad():
        features = dino_model.forward_features(tensor)  # [B, seq_len, dim]
        features = features[:, 0]  # token CLS
        features = features.float()
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    return features


def predict_dino_knn(dino_model, ckpt, pil_img):
    """Prédit lat/lon/pays via KNN sur les features DINOv2."""
    
    image_size = ckpt.get("image_size", 224)
    temperature = ckpt.get("temperature", 0.05)
    best_k = ckpt.get("best_k", 3)
    bank_features = ckpt["bank_features"].float()
    bank_targets  = ckpt["bank_targets"].float()
    bank_countries = ckpt["bank_countries"]
    
    # Normalisation DINOv2
    transform_dino = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform_dino(pil_img.convert("RGB")).unsqueeze(0)
    
    # Features
    query_features = extract_dino_features(dino_model, tensor)  # [1, D]
    
    # Normalise bank
    bank_norm = bank_features / (bank_features.norm(dim=-1, keepdim=True) + 1e-8)
    query_norm = query_features / (query_features.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Similarités cosinus
    similarities = query_norm @ bank_norm.T  # [1, N_bank]
    
    k = min(best_k, bank_norm.shape[0])
    scores, indices = similarities[0].topk(k)
    
    # Pondération par température
    raw_weights = torch.softmax(scores / max(temperature, 1e-3), dim=0)
    
    # Contrainte pays — trouver le pays dominant
    country_weights = {}
    for i, idx in enumerate(indices.tolist()):
        c = bank_countries[idx]
        country_weights[c] = country_weights.get(c, 0.0) + float(raw_weights[i])
    winner_country = max(country_weights, key=country_weights.get)
    confidence = country_weights[winner_country] * 100
    
    # Filtrer par pays gagnant
    mask = torch.tensor([bank_countries[i] == winner_country for i in indices.tolist()])
    filtered_weights = raw_weights * mask.float()
    filtered_weights = filtered_weights / filtered_weights.sum().clamp_min(1e-8)
    
    # Prédiction GPS — moyenne pondérée des vecteurs cibles
    neighbor_targets = bank_targets[indices]  # [k, 3]
    pred_vec = (filtered_weights.unsqueeze(-1) * neighbor_targets).sum(dim=0)
    pred_vec = pred_vec / (pred_vec.norm() + 1e-8)
    
    # Vecteur 3D → lat/lon
    lat, lon = unitvec_to_latlon(pred_vec.unsqueeze(0))
    
    return {
        "country": winner_country,
        "confidence": confidence,
        "lat": lat,
        "lon": lon,
    }


with st.sidebar:
    st.markdown("## GeoGuessrIA")
    st.markdown('<div class="info-tag">CentraleSupélec 2025/2026</div>', unsafe_allow_html=True)
    st.markdown("---")
 
    MENU = {
        "1. Présentation du projet":        "home",
        "2. Dataset":        "dataset",
        "3. Description des modèles":        "models",
        "4. Prédiction":     "predict",
        "5. Analyse du dataset":     "analyse",
    }
    choice = st.radio("Navigation", list(MENU.keys()), label_visibility="collapsed")
    page = MENU[choice]
 
    st.markdown("---")
    st.markdown('<div class="info-tag">Maëlys Hanoire, Diane Verberq, Fanny Badoules</div>', unsafe_allow_html=True)



if page == "home":
    st.markdown("# GeoGuessrIA")
    st.markdown("### Localisation géographique d'images par apprentissage profond")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Dataset</div>
            <div class="value">OSV5M</div>
            <div class="sub">images géoréférencées</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Couverture</div>
            <div class="value">225</div>
            <div class="sub">pays et territoires</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Modèles</div>
            <div class="value">7+</div>
            <div class="sub">architectures testées</div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
 
    st.markdown("#### Objectif")
    st.markdown("""
    À partir d'une image Street View, estimer automatiquement :
    - **Le pays** où la photo a été prise (classification)
    - **Les coordonnées GPS** précises (régression)
 
    Ce projet explore plusieurs architectures CNN et Transformers, avec et sans modules d'attention (CBAM),
    entraînées sur le dataset OSV5M publié à CVPR 2024.
    """)
 
    st.markdown("#### Pipeline")
    st.markdown("""
    ```
    Image Street View → Backbone CNN/ViT → Tête Classification (pays)
                                         → Tête Régression (lat/lon sin/cos)
    ```
    """)



     
elif page == "dataset":
    st.markdown("# Dataset — OSV5M")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
 
    st.markdown("""
    **OpenStreetView-5M** est le premier benchmark open-source de géolocalisation d'images Street View à grande échelle,
    publié à CVPR 2024. Il contient plus de **5.1 millions d'images géoréférencées** couvrant 225 pays.
 
    ##### Caractéristiques
    - Séparation stricte train/test (distance minimale de 1km entre les images)
    - EdA
    """)
 
 
elif page == "models":
    st.markdown("# Modèles entraînés")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
 
    models_info = [
        {
            "name": "ResNet50 + CBAM",
            "tag": "cbam_v1 / cbam_v2",
            "desc": "ResNet50 pretrained ImageNet avec modules CBAM (Channel + Spatial Attention) dans chaque bloc Bottleneck. Architecture multitâche : classification pays + régression GPS sin/cos conditionnée par les probabilités pays.",
            "output": "Pays + GPS",
            "innovation": "GPS conditionné par pays, curriculum lambda, embed_detach",
        },
        {
            "name": "ResNet50 Classif+Reg",
            "tag": "classif_reg",
            "desc": "ResNet50 sans CBAM avec deux têtes : classification pays et régression GPS. Curriculum lambda pour équilibrer les deux losses.",
            "output": "Pays + GPS",
            "innovation": "Baseline de comparaison avec CBAM",
        },
        {
            "name": "ResNet50 Classif Cellules",
            "tag": "classif_cells",
            "desc": "ResNet50 classifiant l'image en cellules géographiques k-means. La prédiction GPS est le centroïde de la cellule prédite. Loss CrossEntropy géographique avec soft labels.",
            "output": "GPS via cellules",
            "innovation": "Soft labels géographiques, WeightedRandomSampler",
        },
        {
            "name": "ResNet50 Geo (régression pure)",
            "tag": "resnet50_geo",
            "desc": "ResNet50 en régression pure sur les coordonnées sin/cos. Hyperparamètres optimisés via Optuna (30 trials).",
            "output": "GPS seulement",
            "innovation": "Optuna hyperparameter search",
        },
        {
            "name": "ResNet18 Geo",
            "tag": "resnet18_geo",
            "desc": "Version légère avec ResNet18 backbone. Plus rapide, moins précis.",
            "output": "GPS seulement",
            "innovation": "Architecture légère",
        },
        {
            "name": "ViT-B/16 Geo",
            "tag": "vit",
            "desc": "Vision Transformer ViT-B/16 avec tête de régression GPS sur vecteur 3D (x, y, z) converti en lat/lon.",
            "output": "GPS seulement",
            "innovation": "Architecture Transformer pure",
        },
        {
            "name": "DINOv2 + KNN",
            "tag": "dino_knn",
            "desc": "DINOv2-large comme extracteur de features, KNN avec pondération par température et contrainte pays. Pas d'entraînement de bout en bout.",
            "output": "Pays + GPS",
            "innovation": "Features self-supervised, KNN géographique",
        },
    ]
 
    for m in models_info:
        with st.expander(f"**{m['name']}** — `{m['tag']}`"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(m["desc"])
                st.markdown(f"**Innovation :** {m['innovation']}")
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Sortie</div>
                    <div class="value" style="font-size:1rem">{m['output']}</div>
                </div>
                """, unsafe_allow_html=True)
 

elif page == "predict":
    st.markdown("# Prédiction géographique")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
 
    MODEL_OPTIONS = {
        "ResNet50 + CBAM v1 (Pays + GPS)":         "cbam_v1",
        "ResNet50 + CBAM v2 more epochs (Pays + GPS)": "cbam_v2",
        "ResNet50 Classif + Reg (Pays + GPS)":        "classif_reg",
        "ResNet50 Classif Cellules (GPS)":           "classif_cells",
        "ResNet50 Geo régression (GPS)":             "resnet50_geo",
        "ResNet18 Geo régression (GPS)":             "resnet18_geo",
        "ViT-B/16 Geo (GPS)":                        "vit",
        "DINOv2 + KNN (Pays + GPS)": "dino_knn",
    }
 
    selected_label = st.selectbox("**Choisir un modèle**", list(MODEL_OPTIONS.keys()))
    selected_model = MODEL_OPTIONS[selected_label]
    st.markdown(f'<div class="model-badge">{selected_label}</div>', unsafe_allow_html=True)
 
    # Upload image
    uploaded = st.file_uploader("**Uploader une image Street View**", type=["jpg", "jpeg", "png"])
 
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        tensor = preprocess(pil_img)
 
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
 
        # Chargement du modèle + prédiction
        with st.spinner("Chargement du modèle et prédiction en cours..."):
            try:
                result = None
                model_inner = None
 
                if selected_model in ("cbam_v1", "cbam_v2"):
                    version = "v1" if selected_model == "cbam_v1" else "v2"
                    model_obj, le = load_cbam(version)
                    model_inner = model_obj
                    result = predict_cbam(model_obj, le, tensor)
 
                elif selected_model == "classif_reg":
                    model_obj, idx_to_country = load_classif_reg()
                    result = predict_classif_reg(model_obj, idx_to_country, tensor)
 
                elif selected_model == "classif_cells":
                    model_obj, centers = load_classif_cells()
                    result = predict_classif_cells(model_obj, centers, tensor)
 
                elif selected_model == "resnet50_geo":
                    model_obj = load_geo_resnet("resnet50")
                    result = predict_geo_resnet(model_obj, tensor)
 
                elif selected_model == "resnet18_geo":
                    model_obj = load_geo_resnet("resnet18")
                    result = predict_geo_resnet(model_obj, tensor)
 
                elif selected_model == "vit":
                    model_obj = load_vit()
                    result = predict_vit(model_obj, tensor)

                elif selected_model == "dino_knn":
                    dino_model, ckpt = load_dino_knn()
                    result = predict_dino_knn(dino_model, ckpt, pil_img)
 
            except Exception as e:
                st.error(f"Erreur lors du chargement ou de la prédiction : {e}")
                result = None
 
        if result:
            col1, col2, col3 = st.columns([1, 1, 1])
 
            with col1:
                st.markdown("##### Image uploadée")
                st.image(pil_img, use_container_width=True)
 
            with col2:
                st.markdown("##### Résultats")
                if result.get("country"):
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:1rem">
                        <div class="label">Pays prédit</div>
                        <div class="value">{result['country']}</div>
                        {f'<div class="sub">Confiance : {result["confidence"]:.1f}%</div>' if result.get("confidence") else ""}
                    </div>
                    """, unsafe_allow_html=True)
 
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:1rem">
                    <div class="label">Latitude</div>
                    <div class="value">{result['lat']:.4f}°</div>
                </div>
                <div class="metric-card">
                    <div class="label">Longitude</div>
                    <div class="value">{result['lon']:.4f}°</div>
                </div>
                """, unsafe_allow_html=True)
 
                if result.get("extra"):
                    st.markdown(f'<div class="info-tag">{result["extra"]}</div>', unsafe_allow_html=True)
 
            with col3:
                st.markdown("##### GradCAM")
                with st.spinner("Génération GradCAM..."):
                    gradcam_img = None

                    if selected_model in ("cbam_v1", "cbam_v2"):
                        gradcam_img = make_gradcam_comparison(pil_img, model_inner, le)

                    elif selected_model == "classif_reg":
                        gradcam_img = make_gradcam_classif_reg(pil_img, model_obj, idx_to_country)

                    elif selected_model == "classif_cells":
                        gradcam_img = make_gradcam_classif_cells(pil_img, model_obj)

                    elif selected_model in ("resnet50_geo", "resnet18_geo"):
                        gradcam_img = make_gradcam_regression(pil_img, model_obj)

                    elif selected_model == "vit":
                        st.info("GradCAM non disponible pour ViT")

                    elif selected_model == "dino_knn":
                        st.info("GradCAM non disponible pour DINOv2 KNN — modèle non-CNN")

                    if gradcam_img is not None:
                        st.image(gradcam_img, use_container_width=True)
                        st.caption("Rouge = zones d'attention forte - Bleu = zones ignorées")
 
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("##### Localisation prédite sur la carte")
 
            m = make_map(result["lat"], result["lon"], result.get("country"))
            st_folium(m, width=None, height=400, returned_objects=[])
 
            # ── Expander : feature maps ────────────────────────────────────────
            if selected_model not in ("dino_knn", "vit"):
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                with st.expander("Visualiser les couches internes du modèle"):
                    try:
                        activations = get_activations(model_obj, tensor)
                        n_layers = len(activations)
                        if n_layers > 0:
                            layer_idx = st.slider(
                                "Couche à visualiser",
                                1, n_layers, min(4, n_layers),
                                format="Couche %d"
                            )
                            act = activations[layer_idx - 1]
                            if act.dim() == 4:
                                import matplotlib.pyplot as plt
                                n_show = min(act.shape[1], 32)
                                cols_per_row = 8
                                rows = max(1, (n_show + cols_per_row - 1) // cols_per_row)
                                fig, axes = plt.subplots(rows, cols_per_row,
                                                         figsize=(cols_per_row * 2, rows * 2))
                                fig.patch.set_facecolor("#0d0d0d")
                                if rows == 1:
                                    axes = [axes]
                                for r in range(rows):
                                    for c in range(cols_per_row):
                                        ch = r * cols_per_row + c
                                        axes[r][c].set_facecolor("#0d0d0d")
                                        if ch < n_show:
                                            axes[r][c].imshow(act[0, ch].numpy(), cmap="inferno")
                                        axes[r][c].axis("off")
                                plt.tight_layout(pad=0.2)
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.markdown(f"`Shape : {list(act.shape)}` — pas de carte spatiale à afficher")
                    except Exception as e:
                        st.warning(f"Visualisation des couches non disponible : {e}")
                        
elif page == "analyse":
    render_analyse_page()


