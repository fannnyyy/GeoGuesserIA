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
import joblib
from cbam import CBAM
from resnet_cbam import GeoGussrAttentionMultiTask
import torch.nn as nn
import torchvision.models as tv_models
import torch.nn.functional as F
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_TRAIN = os.path.join(BASE_DIR, "../dataset_OSV5M/datasets/osv5m/metadata_filtered/rest_filtered_v2.csv")
CSV_TEST  = os.path.join(BASE_DIR, "../dataset_OSV5M/datasets/osv5m/metadata_filtered/samples_filtered_v2.csv")
IMG_DIR   = os.path.join(BASE_DIR, "../dataset_OSV5M/datasets/osv5m/images")

# Mapping MODIS
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


@st.cache_data
def load_csv():
    dfs = []
    for path, split in [(CSV_TRAIN, "rest"), (CSV_TEST, "sample")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["split"] = split
            dfs.append(df)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df["land_cover_name"] = df["land_cover"].map(LAND_COVER_NAMES).fillna("Unknown")
    return df



@st.cache_data
def build_image_index():
    index = {}
    for split_dir in ["samples", "rest_images"]:
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


def render_land_cover(df):
    st.markdown("### Distribution des types de terrain (Land Cover)")

    split_choice = st.radio(
        "Différentes portions du dataset OSV5M utilisé pour les entraînements à analyser",
        ["Partition principal", "Partition échantillonnée"],
        horizontal=True
    )
    if split_choice == "Partition principal":
        df_plot = df[df["split"] == "rest"]
    else:
        df_plot = df[df["split"] == "sample"]

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
        title=f"Distribution land_cover, {len(df_plot):,} images",
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

    top_class = counts.loc[counts["count"].idxmax()]

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


def render_road_index(df):
    st.markdown("### Road Index, densité routière")

    col1, col2 = st.columns(2)

    with col1:
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

    sys.path.insert(0, os.path.join(BASE_DIR, "../model/cnn/cnn_with_attention_module"))
    
    le = joblib.load(os.path.join(BASE_DIR, "../model/saved/label_encoder_v2.pkl"))
    model = GeoGussrAttentionMultiTask(len(le.classes_))
    state = torch.load(
        os.path.join(BASE_DIR, "../model/saved/geoguessr_model_attention_classif_reg_v2.pt"),
        map_location="cpu", weights_only=False
    )
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    return _extract_features_with_hook(model, "avgpool", n_samples, _df)


@st.cache_data
def extract_resnet_geo_embeddings_for_tsne(variant="resnet50", n_samples=5000, _df=None):
    """Pour GeoResNet (régression pure), hook sur backbone.layer4[-1]"""

    key = "resnet50_geo_final" if variant == "resnet50" else "resnet18_geo"
    ckpt = torch.load(
        os.path.join(BASE_DIR, f"../model/saved/{key}.pt"),
        map_location="cpu", weights_only=False
    )
    p = ckpt.get("hyperparams", {"backbone": variant, "hidden_dim": 512, "n_layers": 1, "dropout_p": 0.4})


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


@st.cache_data
def extract_resnet_classif_embeddings_for_tsne(n_samples=5000, _df=None):
    """Pour GeoResNetClassif (classification cellules k-means), hook sur backbone.layer4[-1]"""

    cells_path = "/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/saved/cells_kmeans.pkl"
    with open(cells_path, "rb") as f:
        cells = pickle.load(f)
    n_cells = cells["n_cells"]

    ckpt = torch.load(
        os.path.join(BASE_DIR, "../model/saved/resnet50_classification.pt"),
        map_location="cpu", weights_only=False
    )
    cfg_model = ckpt.get("config", {"hidden_dim": 512, "dropout_p": 0.4})

    model = GeoResNetClassif(
        n_cells=n_cells,
        hidden_dim=cfg_model.get("hidden_dim", 512),
        dropout_p=cfg_model.get("dropout_p", 0.4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return _extract_features_with_hook(model, "backbone.layer4", n_samples, _df)



TSNE_CACHE_DIR = os.path.join(BASE_DIR, "../model/saved/tsne_cache")

def compute_tsne(features, n_iter=1000):
    """Calcule le t-SNE sur les features."""
    features = StandardScaler().fit_transform(features)

    # Réduction PCA d'abord pour accélérer
    n_components = min(20, features.shape[1], features.shape[0] - 1)
    features_pca = PCA(n_components=n_components).fit_transform(features)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=n_iter,
        random_state=42,
    )
    return tsne.fit_transform(features_pca)

def compute_tsne_cached(features, countries, model_name):
    os.makedirs(TSNE_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(TSNE_CACHE_DIR, f"tsne_{model_name}_p30.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        return data["embedding"], list(data["countries"])

    embedding = compute_tsne(features)

    np.savez(cache_file, embedding=embedding, countries=np.array(countries))

    return embedding, countries

def render_tsne(df):
    st.markdown("### t-SNE des embeddings")

    model_choice = st.selectbox(
        "Modèle",
        [
            "DINOv2 KNN",
            "ResNet50 + CBAM",
            "ResNet50 Classif Cellules (classification pure)",
            "ResNet50 Reg (régression pure)",
        ],
    )

    col_launch, col_reset = st.columns([3, 1])
    with col_launch:
        launch = st.button("Lancer le t-SNE", type="primary")
    with col_reset:
        force_recompute = st.button("Vider le cache")


    MODEL_NAME_MAP = {
        "DINOv2 KNN": "dino",
        "ResNet50 + CBAM": "cbam",
        "ResNet50 Classif Cellules (classification pure)": "resnet_classif",
        "ResNet50 Reg (régression pure)": "resnet_reg",
    }

    model_name = MODEL_NAME_MAP.get(model_choice, model_choice.split(" ")[0].lower())
    cache_file = os.path.join(TSNE_CACHE_DIR, f"tsne_{model_name}_p30.npz")

    if force_recompute and os.path.exists(cache_file):
        os.remove(cache_file)
        st.success("Cache supprimé !")

    if launch :
        with st.spinner(f"Extraction des features et calcul du t-SNE sur {N_TSNE_SAMPLES} images..."):

            if "DINO" in model_choice:
                features, countries, _ = load_dino_embeddings_for_tsne(N_TSNE_SAMPLES)
            
            elif "CBAM" in model_choice:
                features, countries = extract_cbam_embeddings_for_tsne(N_TSNE_SAMPLES, _df=df)
            
            elif "ResNet50 Reg" in model_choice:
                features, countries = extract_resnet_geo_embeddings_for_tsne("resnet50", N_TSNE_SAMPLES, _df=df)
            
            elif "Classif" in model_choice:
                features, countries = extract_resnet_classif_embeddings_for_tsne(N_TSNE_SAMPLES, _df=df)

            with st.spinner("Calcul t-SNE en cours..."):
                embedding, countries = compute_tsne_cached(features, countries, model_name)

            tsne_df = pd.DataFrame({
                "x": embedding[:, 0],
                "y": embedding[:, 1],
                "country": countries[:len(embedding)],
            })

            top15 = tsne_df["country"].value_counts().head(15).index.tolist()
            tsne_df["label"] = tsne_df["country"].apply(
                lambda c: c if c in top15 else "Autres"
            )
            fig = px.scatter(
                tsne_df,
                x="x", y="y",
                color="label",
                title=f"t-SNE, {model_choice}, colorié par pays (Top 15)",
                opacity=0.6,
                size_max=4,
                labels={"x": "t-SNE 1", "y": "t-SNE 2", "label": "Pays"},
            )

            fig.update_traces(marker=dict(size=3))
            fig.update_layout(
                plot_bgcolor="#0d0d0d",
                paper_bgcolor="#0d0d0d",
                font_color="#f0ede8",
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig, use_container_width=True)




def render_analyse_page():
    st.markdown("# Analyse du dataset OSV5M")
    st.markdown("""
    Cette page explore les caractéristiques du dataset.
    """)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Chargement CSV
    with st.spinner("Chargement des métadonnées..."):
        df = load_csv()

    if df is None:
        st.error("Fichiers CSV non trouvés. Vérifiez les chemins dans CSV_TRAIN et CSV_TEST.")
        return


    tab1, tab2, tab3 = st.tabs([
        "Couverture du sol",
        "Index routier",
        "t-SNE",
    ])

    with tab1:
        render_land_cover(df)

    with tab2:
        render_road_index(df)

    with tab3:
        render_tsne(df)