"""
Page 5 — Analyse du dataset OSV5M
À intégrer dans streamlit_app.py comme nouvelle page dans le menu.

Contenu :
    1. Distribution land_cover (barplot + noms lisibles)
    2. Road index (distribution + corrélation avec erreur GPS)
    3. t-SNE des embeddings (DINO / CBAM / ResNet)
       → coloré par pays (top 15) ou par land_cover

Prérequis :
    pip install scikit-learn plotly
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from torchvision import transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_TRAIN = os.path.join(BASE_DIR, "../dataset_OSV5M/datasets/osv5m/metadata_filtered/train_filtered.csv")
CSV_TEST  = os.path.join(BASE_DIR, "../dataset_OSV5M/datasets/osv5m/metadata_filtered/test_filtered.csv")
IMG_DIR   = os.path.join(BASE_DIR, "../dataset_OSV5M/datasets/osv5m/images")

# Mapping MODIS Land Cover
LAND_COVER_NAMES = {
    0:  "Water",
    1:  "Evergreen Needleleaf Forest",
    2:  "Evergreen Broadleaf Forest",
    3:  "Deciduous Needleleaf Forest",
    4:  "Deciduous Broadleaf Forest",
    5:  "Mixed Forest",
    6:  "Closed Shrublands",
    7:  "Open Shrublands",
    8:  "Woody Savannas",
    9:  "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
}

LAND_COVER_COLORS = {
    0:  "#3498db",   # bleu eau
    1:  "#1a5c2a",   # vert foncé
    2:  "#27ae60",   # vert tropical
    3:  "#2ecc71",   # vert clair
    4:  "#f39c12",   # orange forêt
    5:  "#e67e22",   # orange mixte
    6:  "#d35400",   # brun shrub fermé
    7:  "#e8c99a",   # beige shrub ouvert
    8:  "#c8a882",   # beige savane boisée
    9:  "#f0e68c",   # jaune savane
    10: "#90ee90",   # vert pâle prairie
    11: "#4682b4",   # bleu foncé zones humides
}

N_TSNE_SAMPLES = 1000

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DES CSV
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_csv():
    dfs = []
    for path, split in [(CSV_TRAIN, "train"), (CSV_TEST, "test")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["split"] = split
            dfs.append(df)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df["land_cover_name"] = df["land_cover"].map(LAND_COVER_NAMES).fillna("Unknown")
    return df


# ─────────────────────────────────────────────────────────────────
# INDEX DES IMAGES
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def build_image_index():
    index = {}
    for split_dir in ["train", "test", "rest_images"]:
        split_path = os.path.join(IMG_DIR, split_dir)
        if not os.path.isdir(split_path):
            continue
        for subdir in os.listdir(split_path):
            subdir_path = os.path.join(split_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in os.listdir(subdir_path):
                if fname.endswith(".jpg"):
                    index[fname[:-4]] = os.path.join(subdir_path, fname)
    return index


# ─────────────────────────────────────────────────────────────────
# SECTION 1 — DISTRIBUTION LAND COVER
# ─────────────────────────────────────────────────────────────────

def render_land_cover(df):
    st.markdown("### Distribution des types de terrain (Land Cover)")
    st.markdown("""
    La colonne `land_cover` du dataset OSV5M utilise la classification **MODIS** — 
    chaque image est associée au type de terrain dominant dans sa zone géographique.
    """)

    # Distribution globale
    split_choice = st.radio(
        "Split à analyser",
        ["Train + Test", "Train seulement", "Test seulement"],
        horizontal=True
    )
    if split_choice == "Train seulement":
        df_plot = df[df["split"] == "train"]
    elif split_choice == "Test seulement":
        df_plot = df[df["split"] == "test"]
    else:
        df_plot = df

    counts = df_plot["land_cover"].value_counts().reset_index()
    counts.columns = ["land_cover", "count"]
    counts["name"] = counts["land_cover"].map(LAND_COVER_NAMES).fillna("Unknown")
    counts["color"] = counts["land_cover"].map(LAND_COVER_COLORS).fillna("#888")
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    counts = counts.sort_values("land_cover")

    fig = px.bar(
        counts,
        x="name",
        y="count",
        color="name",
        color_discrete_map={row["name"]: row["color"] for _, row in counts.iterrows()},
        text=counts["pct"].astype(str) + "%",
        labels={"name": "Type de terrain", "count": "Nombre d'images"},
        title=f"Distribution land_cover — {len(df_plot):,} images",
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#0d0d0d",
        paper_bgcolor="#0d0d0d",
        font_color="#f0ede8",
        xaxis_tickangle=-30,
        title_font_size=14,
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Observation clé
    top_class = counts.loc[counts["count"].idxmax()]
    st.info(
        f"**Observation** — La classe dominante est **{top_class['name']}** "
        f"({top_class['pct']}% des images). Les forêts (classes 1-5) représentent "
        f"{counts[counts['land_cover'].isin([1,2,3,4,5])]['pct'].sum():.1f}% du dataset — "
        f"des zones avec peu d'indices géographiques discriminants (pas de panneaux, peu d'infrastructures)."
    )

    # Distribution par pays top 10
    st.markdown("#### Land cover par pays (Top 10 pays)")
    top_countries = df_plot["country"].value_counts().head(10).index.tolist()
    df_top = df_plot[df_plot["country"].isin(top_countries)]
    pivot = df_top.groupby(["country", "land_cover_name"]).size().reset_index(name="count")

    fig2 = px.bar(
        pivot,
        x="country",
        y="count",
        color="land_cover_name",
        barmode="stack",
        labels={"country": "Pays", "count": "Nombre d'images", "land_cover_name": "Type terrain"},
        title="Composition land_cover par pays (Top 10)",
    )
    fig2.update_layout(
        plot_bgcolor="#0d0d0d",
        paper_bgcolor="#0d0d0d",
        font_color="#f0ede8",
        title_font_size=14,
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# SECTION 2 — ROAD INDEX
# ─────────────────────────────────────────────────────────────────

def render_road_index(df):
    st.markdown("### Road Index — densité routière")
    st.markdown("""
    Le `road_index` mesure la densité du réseau routier autour du point GPS de l'image.
    Un index élevé indique une zone urbaine bien connectée, un index faible indique une zone isolée.
    **Hypothèse** : les images avec un road_index faible ont moins d'indices visuels (routes vides, 
    pas de panneaux) et sont donc plus difficiles à géolocaliser.
    """)

    col1, col2 = st.columns(2)

    with col1:
        # Distribution road_index
        sample = df["road_index"].dropna().sample(min(50000, len(df)), random_state=42)
        fig = px.histogram(
            x=sample,
            nbins=50,
            labels={"x": "Road Index", "y": "Nombre d'images"},
            title="Distribution du Road Index",
            color_discrete_sequence=["#c8a882"],
        )
        fig.update_layout(
            plot_bgcolor="#0d0d0d",
            paper_bgcolor="#0d0d0d",
            font_color="#f0ede8",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Road index par land_cover
        df_sample = df.dropna(subset=["road_index", "land_cover"]).sample(
            min(50000, len(df)), random_state=42
        )
        df_sample["land_cover_name"] = df_sample["land_cover"].map(LAND_COVER_NAMES).fillna("Unknown")
        median_by_lc = df_sample.groupby("land_cover_name")["road_index"].median().reset_index()
        median_by_lc = median_by_lc.sort_values("road_index", ascending=True)

        fig2 = px.bar(
            median_by_lc,
            x="road_index",
            y="land_cover_name",
            orientation="h",
            labels={"road_index": "Road Index médian", "land_cover_name": "Type terrain"},
            title="Road Index médian par type de terrain",
            color="road_index",
            color_continuous_scale="Oranges",
        )
        fig2.update_layout(
            plot_bgcolor="#0d0d0d",
            paper_bgcolor="#0d0d0d",
            font_color="#f0ede8",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Stats rapides
    col1, col2, col3 = st.columns(3)
    with col1:
        pct_low = (df["road_index"] < 3).mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Road Index &lt; 3</div>
            <div class="value">{pct_low:.1f}%</div>
            <div class="sub">zones peu connectées</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        pct_mid = ((df["road_index"] >= 3) & (df["road_index"] < 5)).mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Road Index 3-5</div>
            <div class="value">{pct_mid:.1f}%</div>
            <div class="sub">zones semi-rurales</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        pct_high = (df["road_index"] >= 5).mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Road Index &gt; 5</div>
            <div class="value">{pct_high:.1f}%</div>
            <div class="sub">zones urbaines</div>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "**Lien avec les performances** — Les zones avec road_index faible correspondent aux "
        "terrains forestiers et ruraux qui dominent le dataset. Ces zones produisent des images "
        "similaires visuellement entre pays différents, ce qui explique les faibles performances "
        "de géolocalisation."
    )


# ─────────────────────────────────────────────────────────────────
# SECTION 3 — t-SNE DES EMBEDDINGS
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_dino_embeddings_for_tsne(n_samples=5000):
    """Charge les embeddings DINO depuis le bank sauvegardé."""
    dino_path = os.path.join(BASE_DIR, "../model/saved/dinov2_knn_geo_index.pt")
    if not os.path.exists(dino_path):
        return None, None, None

    ckpt = torch.load(dino_path, map_location="cpu", weights_only=False)
    bank_features  = ckpt["bank_features"].float()
    bank_countries = ckpt["bank_countries"]
    bank_ids       = ckpt["bank_ids"]

    # Sous-échantillonnage aléatoire
    n = min(n_samples, len(bank_features))
    idx = random.sample(range(len(bank_features)), n)

    features  = bank_features[idx].numpy()
    countries = [bank_countries[i] for i in idx]
    ids       = [bank_ids[i] for i in idx]

    return features, countries, ids

@st.cache_data
def extract_cbam_embeddings_for_tsne(n_samples=5000, _df=None):
    import joblib
    sys.path.insert(0, os.path.join(BASE_DIR, "../model/cnn/cnn_with_attention_module"))
    from cbam import CBAM
    from resnet_cbam import GeoGussrAttentionMultiTask

    le = joblib.load(os.path.join(BASE_DIR, "../model/saved/label_encoder.pkl"))
    model = GeoGussrAttentionMultiTask(len(le.classes_))
    state = torch.load(
        os.path.join(BASE_DIR, "../model/saved/geoguessr_model_attention_classif_reg.pt"),
        map_location="cpu", weights_only=False
    )
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    return _extract_features_with_hook(model, "avgpool", n_samples, _df)


@st.cache_data
def extract_resnet_geo_embeddings_for_tsne(variant="resnet50", n_samples=5000, _df=None):
    """Pour GeoResNet (régression pure) — hook sur backbone.layer4[-1]"""
    key = "resnet50_geo_final" if variant == "resnet50" else "resnet18_geo"
    ckpt = torch.load(
        os.path.join(BASE_DIR, f"../model/saved/{key}.pt"),
        map_location="cpu", weights_only=False
    )
    p = ckpt.get("hyperparams", {"backbone": variant, "hidden_dim": 512, "n_layers": 1, "dropout_p": 0.4})

    import torch.nn as nn
    import torchvision.models as tv_models
    import torch.nn.functional as F

    class GeoResNet(nn.Module):
        def __init__(self, backbone="resnet50", hidden_dim=512, n_layers=1, dropout_p=0.4):
            super().__init__()
            if backbone == "resnet50":
                self.backbone = tv_models.resnet50(weights=None)
            else:
                self.backbone = tv_models.resnet18(weights=None)
            in_features = self.backbone.fc.in_features
            head = nn.Sequential(
                nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, 4),
            )
            self.backbone.fc = head
        def forward(self, x):
            x = self.backbone(x)
            lat = F.normalize(x[:, 0:2], dim=1)
            lon = F.normalize(x[:, 2:4], dim=1)
            return torch.cat([lat, lon], dim=1)

    model = GeoResNet(
        backbone=p.get("backbone", variant),
        hidden_dim=p.get("hidden_dim", 512),
        n_layers=p.get("n_layers", 1),
        dropout_p=p.get("dropout_p", 0.4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return _extract_features_with_hook(model, "backbone.avgpool", n_samples, _df)


def _extract_features_with_hook(model, layer_path, n_samples, df):
    """Extrait les features via hook sur la couche spécifiée."""
    if df is None:
        return None, None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Récupérer le module via le chemin
    module = model
    for attr in layer_path.split("."):
        module = getattr(module, attr)

    features_list = []
    def hook_fn(m, inp, out):
        features_list.append(out.detach().cpu().flatten(1))

    hook = module.register_forward_hook(hook_fn)

    image_index = build_image_index()
    sample_df = df.dropna(subset=["latitude", "longitude"]).sample(
        min(n_samples * 2, len(df)), random_state=42
    )

    countries = []
    with torch.no_grad():
        for _, row in sample_df.iterrows():
            if len(features_list) >= n_samples:
                break
            img_id = str(row["id"])
            if img_id not in image_index:
                continue
            try:
                img = Image.open(image_index[img_id]).convert("RGB")
                tensor = transform(img).unsqueeze(0)
                model(tensor)
                countries.append(str(row.get("country", "UNK")))
            except Exception:
                if features_list:
                    features_list.pop()

    hook.remove()

    if not features_list:
        return None, None

    features = torch.cat(features_list, dim=0).numpy()
    return features, countries

TSNE_CACHE_DIR = os.path.join(BASE_DIR, "../model/saved/tsne_cache")

def compute_tsne_cached(features, countries, model_name, perplexity=30):
    """Calcule le t-SNE et sauvegarde sur disque."""
    os.makedirs(TSNE_CACHE_DIR, exist_ok=True)
    
    # Nom de fichier unique selon le modèle et la perplexité
    cache_file = os.path.join(TSNE_CACHE_DIR, f"tsne_{model_name}_p{perplexity}.npz")
    
    # Si le cache existe → charger directement
    if os.path.exists(cache_file):
        st.info("✅ t-SNE chargé depuis le cache disque")
        data = np.load(cache_file, allow_pickle=True)
        return data["embedding"], list(data["countries"])
    
    # Sinon → calculer et sauvegarder
    st.info("⏳ Premier calcul — sera mis en cache pour les prochaines fois")
    embedding =  compute_tsne_cached(features, countries, model_name, perplexity)
    
    np.savez(
        cache_file,
        embedding=embedding,
        countries=np.array(countries),
    )
    st.success(f"✅ t-SNE sauvegardé dans {cache_file}")
    
    return embedding, countries


def render_tsne(df):
    st.markdown("### t-SNE des embeddings")
    st.markdown("""
    Le t-SNE projette les embeddings haute dimension (features extraites par les modèles) 
    en 2D pour visualiser la structure des représentations apprises. Si les clusters 
    correspondent aux pays → le modèle discrimine bien géographiquement. Sinon → les 
    features sont trop génériques.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        model_choice = st.selectbox(
            "Modèle",
            [
                "DINOv2 KNN (bank pré-calculé)",
                "ResNet50 + CBAM v1",
                "ResNet50 Geo (régression pure)",
                "ResNet18 Geo (régression pure)",
            ],
        )
    with col2:
        color_by = st.selectbox(
            "Colorier par",
            ["Pays (Top 15)", "Land Cover"],
        )
    with col3:
        perplexity = st.slider("Perplexité t-SNE", 5, 50, 30)

    col_launch, col_reset = st.columns([3, 1])
    with col_launch:
        launch = st.button("🔄 Lancer le t-SNE", type="primary")
    with col_reset:
        force_recompute = st.button("🗑️ Vider le cache")

    # Nom du modèle pour le cache
    model_name = model_choice.split(" ")[0].lower()  # "dino", "resnet50", etc.
    cache_file = os.path.join(TSNE_CACHE_DIR, f"tsne_{model_name}_p{perplexity}.npz")

    if force_recompute and os.path.exists(cache_file):
        os.remove(cache_file)
        st.success("Cache supprimé !")

    if launch :
        with st.spinner(f"Extraction des features et calcul du t-SNE sur {N_TSNE_SAMPLES} images..."):

            if "DINO" in model_choice:
                features, countries, _ = load_dino_embeddings_for_tsne(N_TSNE_SAMPLES)
            
            elif "CBAM" in model_choice:
                features, countries = extract_cbam_embeddings_for_tsne(N_TSNE_SAMPLES, _df=df)
            
            elif "ResNet50 Geo" in model_choice:
                features, countries = extract_resnet_geo_embeddings_for_tsne("resnet50", N_TSNE_SAMPLES, _df=df)
            
            elif "ResNet18" in model_choice:
                features, countries = extract_resnet_geo_embeddings_for_tsne("resnet18", N_TSNE_SAMPLES, _df=df)

            # Récupérer land_cover depuis le CSV si nécessaire
            if color_by == "Land Cover" and df is not None:
                id_to_lc = dict(zip(df["id"].astype(str), df["land_cover"]))

            # Calcul t-SNE
            with st.spinner("Calcul t-SNE en cours (peut prendre 1-3 minutes)..."):
                embedding = compute_tsne_cached(features, countries, model_name, perplexity)

            # Préparation du plot
            tsne_df = pd.DataFrame({
                "x": embedding[:, 0],
                "y": embedding[:, 1],
                "country": countries[:len(embedding)],
            })

            if color_by == "Pays (Top 15)":
                top15 = tsne_df["country"].value_counts().head(15).index.tolist()
                tsne_df["label"] = tsne_df["country"].apply(
                    lambda c: c if c in top15 else "Autres"
                )
                fig = px.scatter(
                    tsne_df,
                    x="x", y="y",
                    color="label",
                    title=f"t-SNE — {model_choice} — colorié par pays (Top 15)",
                    opacity=0.6,
                    size_max=4,
                    labels={"x": "t-SNE 1", "y": "t-SNE 2", "label": "Pays"},
                )

            else:  # Land Cover
                tsne_df["land_cover_name"] = tsne_df["country"].map(
                    lambda c: "Unknown"
                )
                # Si on a les land covers du bank DINO
                fig = px.scatter(
                    tsne_df,
                    x="x", y="y",
                    color="country",
                    title=f"t-SNE — {model_choice}",
                    opacity=0.6,
                    labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                )

            fig.update_traces(marker=dict(size=3))
            fig.update_layout(
                plot_bgcolor="#0d0d0d",
                paper_bgcolor="#0d0d0d",
                font_color="#f0ede8",
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Interprétation
            st.markdown("#### Interprétation")
            st.markdown("""
            - **Clusters bien séparés par pays** → le modèle discrimine géographiquement ✅
            - **Clusters mélangés** → les features sont trop génériques, certains pays 
              se ressemblent visuellement → difficulté de géolocalisation ⚠️
            - **Regroupements inattendus** → révèle des biais dans le dataset (ex: 
              images US trop nombreuses formant un méga-cluster)
            """)


# ─────────────────────────────────────────────────────────────────
# RENDER PRINCIPAL — à appeler dans streamlit_app.py
# ─────────────────────────────────────────────────────────────────

def render_analyse_page():
    st.markdown("# Analyse du dataset OSV5M")
    st.markdown("""
    Cette page explore les caractéristiques du dataset pour comprendre pourquoi 
    les modèles de géolocalisation ont des performances limitées.
    """)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Chargement CSV
    with st.spinner("Chargement des métadonnées..."):
        df = load_csv()

    if df is None:
        st.error("Fichiers CSV non trouvés. Vérifiez les chemins dans CSV_TRAIN et CSV_TEST.")
        return

    # Stats rapides en haut
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Images totales</div>
            <div class="value">{len(df):,}</div>
            <div class="sub">train + test</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Pays couverts</div>
            <div class="value">{df['country'].nunique()}</div>
            <div class="sub">pays distincts</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Types terrain</div>
            <div class="value">{df['land_cover'].nunique()}</div>
            <div class="sub">classes MODIS</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        road_mean = df["road_index"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Road Index moyen</div>
            <div class="value">{road_mean:.2f}</div>
            <div class="sub">densité routière</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Tabs pour les 3 sections
    tab1, tab2, tab3 = st.tabs([
        "🌿 Land Cover",
        "🛣️ Road Index",
        "🔵 t-SNE Embeddings",
    ])

    with tab1:
        render_land_cover(df)

    with tab2:
        render_road_index(df)

    with tab3:
        render_tsne(df)