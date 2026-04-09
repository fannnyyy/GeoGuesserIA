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
import timm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CBAM_SRC = os.path.join(BASE_DIR, "../model/cnn/cnn_with_attention_module")
if CBAM_SRC not in sys.path:
    sys.path.insert(0, CBAM_SRC)

from cbam import CBAM
from resnet_cbam import GeoGussrAttentionMultiTask
from gradcam_streamlit import (
    make_gradcam_comparison,
    make_gradcam_classif_reg,
    make_gradcam_classif_cells,
    make_gradcam_regression,
)
from page_analyse_streamlit import render_analyse_page
from prediction_streamlit import (
    GeoResNet,
    GeoResNetClassif,
    GeoResNetClassifRegress,
    ViTGeo,
    normalize_vec,
    unitvec_to_latlon,
    sincos_to_latlon,
    preprocess,
    load_cbam,
    load_classif_reg,
    load_classif_cells,
    load_geo_resnet,
    load_vit,
    load_dino_knn,
    predict_cbam,
    predict_classif_reg,
    predict_classif_cells,
    predict_geo_resnet,
    predict_vit,
    predict_dino_knn,
    extract_dino_features,
    make_map,
    get_activations,
    get_activations_resnet,
)


PATHS = {
    "cbam_v2": {
        "pt":  os.path.join(BASE_DIR, "../model/saved/geoguessr_model_attention_classif_reg_v2.pt"),
        "pkl": os.path.join(BASE_DIR, "../model/saved/label_encoder_v2.pkl"),
        "src": os.path.join(BASE_DIR, "../model/saved"),
    },
    # ResNet50 classif cellules k-means
    "classif_cells": {
        "pt":   os.path.join(BASE_DIR, "../model/saved/resnet50_classification_210k.pt"),
        "pkl":  None,
        "cells": os.path.join("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/resnet/resnet_full_classif/cells_kmeans.pkl"),
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
    # ResNet50 classif + reg
    "classif_reg": {
        "pt":  os.path.join(BASE_DIR, "../model/saved/resnet50_classif_regress.pt"),
        "pkl": None,
    },
    # ResNet50 geo final (régression pure)
    "resnet50_geo": {
        "pt": os.path.join(BASE_DIR, "../model/saved/resnet50_geo_final.pt"),
    },
    # ResNet18 geo
    "resnet18_geo": {
        "pt": os.path.join(BASE_DIR, "../model/saved/resnet18_geo.pt"),
    },
}


with st.sidebar:
    st.markdown("## GeoGuessrIA")
    st.markdown('<div class="info-tag">CentraleSupélec 2025/2026</div>', unsafe_allow_html=True)
    st.markdown("---")
 
    MENU = {
        "1. Présentation du projet": "home",
        "2. Analyse du dataset": "analyse",
        "3. Description des modèles": "models",
        "4. Prédiction": "predict",
    }
    choice = st.radio("Navigation", list(MENU.keys()), label_visibility="collapsed")
    page = MENU[choice]
 
    st.markdown("---")
    st.markdown('<div class="info-tag">Maëlys Hanoire, \tDiane Verberq, \tFanny Badoules</div>', unsafe_allow_html=True)



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
    MODEL_FAMILIES = {
        "CNN classiques": {
            "ResNet50 Classif Cellules (GPS)":    "classif_cells",
            "ResNet50 Classif + Reg (Pays + GPS)": "classif_reg",
            "ResNet50 Geo régression (GPS)":      "resnet50_geo",
            "ResNet18 Geo régression (GPS)":      "resnet18_geo",
        },
        "CNN avec attention (CBAM)": {
            "ResNet50 + CBAM (Pays + GPS)": "cbam_v2",
        },
        "Transformers": {
            "ViT-B/16 Geo (GPS)": "vit",
        },
        "Self-supervised + KNN": {
            "DINOv2 + KNN (Pays + GPS)": "dino_knn",
        },
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        family = st.selectbox("**Famille de modèles**", list(MODEL_FAMILIES.keys()))
    with col2:
        selected_label = st.selectbox("**Modèle**", list(MODEL_FAMILIES[family].keys()))

    selected_model = MODEL_FAMILIES[family][selected_label]
    st.markdown(f'<div class="model-badge">{family} — {selected_label}</div>', unsafe_allow_html=True)
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
 
                if selected_model in ("cbam_v2"):
                    version = "v2"
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

                    if selected_model in ("cbam_v2"):
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
 

            if selected_model not in ("dino_knn", "vit"):
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                with st.expander("Visualiser les couches internes du modèle"):
                    if selected_model in ("resnet50_geo", "resnet18_geo"):
                        activations = get_activations_resnet(model_obj, tensor)
                    else:
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
                    
                        
elif page == "analyse":
    render_analyse_page()


