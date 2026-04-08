"""
EDA complet — Dataset OSV5M (GeoGuessr IA)
==========================================
Lance ce script depuis la racine du projet :
    python eda_osv5m.py

Sorties générées :
    eda_outputs/
        01_world_heatmap.html          ← carte interactive Folium (densité mondiale)
        02_continent_distribution.png
        03_country_top30.png
        04_lat_lon_histograms.png
        05_cell_distribution.png       ← distribution des cellules k-means
        06_cell_sizes.png              ← nbre d'images par cellule
        07_image_resolution.png        ← si les métadonnées contiennent w/h
        08_missing_values.png
        09_coordinate_kde.png          ← densité lat/lon en 2D
        10_haversine_intra_cell.png    ← dispersion intra-cellule
        eda_report.txt                 ← résumé textuel complet
"""

import os
import math
import pickle
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIG  — adapte ces chemins à ton arborescence
# ──────────────────────────────────────────────────────────────
BASE    = Path("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m")
TRAIN_CSV   = BASE / "metadata_filtered/rest_filtered_v2.csv"
TEST_CSV    = BASE / "metadata_filtered/samples_filtered_v2.csv"
CELLS_PKL   = Path("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/resnet/resnet_full_classif/cells_kmeans.pkl")
OUT_DIR     = Path("eda_outputs_test")
OUT_DIR.mkdir(exist_ok=True)

STYLE = "seaborn-v0_8-whitegrid"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

# ──────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ──────────────────────────────────────────────────────────────
print("Chargement des CSV…")
train = pd.read_csv(TRAIN_CSV, low_memory=False)
test  = pd.read_csv(TEST_CSV,  low_memory=False)

train["split"] = "train"
test["split"]  = "test"
df = pd.concat([train, test], ignore_index=True)

print(f"  Train : {len(train):,} lignes  |  Test : {len(test):,} lignes")
print(f"  Colonnes : {df.columns.tolist()}\n")

# ──────────────────────────────────────────────────────────────
# 2. RAPPORT TEXTE — stats de base
# ──────────────────────────────────────────────────────────────
report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(msg)

log("=" * 60)
log("EDA OSV5M — RAPPORT")
log("=" * 60)
log(f"\nTrain : {len(train):,}   Test : {len(test):,}   Total : {len(df):,}")
log(f"Colonnes ({len(df.columns)}) : {', '.join(df.columns)}")

# Valeurs manquantes
log("\n── Valeurs manquantes (%) ──────────────────────────────────")
missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
for col, pct in missing[missing > 0].items():
    log(f"  {col:<30} {pct:6.2f}%")

# Stats coordonnées
log("\n── Coordonnées GPS ─────────────────────────────────────────")
for col in ["latitude", "longitude"]:
    s = df[col].dropna()
    log(f"  {col}: min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}  std={s.std():.4f}")

# ──────────────────────────────────────────────────────────────
# 3. CARTE INTERACTIVE FOLIUM
# ──────────────────────────────────────────────────────────────
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    print("Génération de la carte Folium…")

    # Sous-échantillon pour performance
    sample = df[["latitude", "longitude", "split"]].dropna().sample(
        n=min(80_000, len(df)), random_state=42
    )

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    # HeatMap densité globale
    heat_data = sample[["latitude", "longitude"]].values.tolist()
    HeatMap(heat_data, radius=4, blur=6, min_opacity=0.3).add_to(m)

    # Marqueurs d'exemples (1 000 points colorés par split)
    colors = {"train": "blue", "test": "red"}
    cluster = MarkerCluster(name="Exemples (1 000)").add_to(m)
    for _, row in sample.sample(1000, random_state=0).iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=colors.get(row["split"], "gray"),
            fill=True,
            fill_opacity=0.7,
            tooltip=f"{row['split']}  ({row['latitude']:.3f}, {row['longitude']:.3f})",
        ).add_to(cluster)

    folium.LayerControl().add_to(m)

    map_path = OUT_DIR / "01_world_heatmap.html"
    m.save(str(map_path))
    log(f"\nCarte interactive sauvegardée : {map_path}")

except ImportError:
    log("\n[WARN] folium non installé — pip install folium  →  carte ignorée")

# ──────────────────────────────────────────────────────────────
# 4. HISTOGRAMMES LAT / LON
# ──────────────────────────────────────────────────────────────
print("Histogrammes lat/lon…")
with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, col, color in zip(axes, ["latitude", "longitude"], ["#4477AA", "#EE7733"]):
        vals = df[col].dropna()
        ax.hist(vals, bins=180, color=color, alpha=0.85, edgecolor="none")
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Nombre d'images")
        ax.set_title(f"Distribution — {col}")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}°"))
    fig.suptitle("Distribution des coordonnées GPS", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_lat_lon_histograms.png")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────
# 5. KDE 2D  latitude × longitude
# ──────────────────────────────────────────────────────────────
print("KDE 2D…")
sample_kde = df[["latitude", "longitude"]].dropna().sample(
    n=min(200_000, len(df)), random_state=1
)
with plt.style.context(STYLE):
    fig, ax = plt.subplots(figsize=(14, 7))
    h = ax.hist2d(
        sample_kde["longitude"], sample_kde["latitude"],
        bins=[360, 180], cmap="YlOrRd",
        norm=matplotlib.colors.LogNorm(),
    )
    plt.colorbar(h[3], ax=ax, label="Nombre d'images (log)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Densité 2D des images dans le monde", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "09_coordinate_kde.png")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────
# 6. DISTRIBUTION PAR PAYS / CONTINENT  (si colonne dispo)
# ──────────────────────────────────────────────────────────────
country_col = next((c for c in df.columns if c.lower() in
                    ["country", "country_iso", "country_code", "pays"]), None)

if country_col:
    print(f"Distribution pays (colonne : {country_col})…")
    top30 = df[country_col].value_counts().head(30)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 7))
        top30.plot(kind="barh", ax=ax, color="#4477AA")
        ax.invert_yaxis()
        ax.set_xlabel("Nombre d'images")
        ax.set_title("Top 30 pays", fontweight="bold")
        for i, v in enumerate(top30):
            ax.text(v + len(df)*0.001, i, f"{v:,}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "03_country_top30.png")
        plt.close(fig)

    log(f"\n── Top 10 pays ─────────────────────────────────────────────")
    for country, cnt in top30.head(10).items():
        log(f"  {str(country):<35} {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")

continent_col = next((c for c in df.columns if c.lower() in
                      ["continent", "region", "zone"]), None)
if continent_col:
    print(f"Distribution continent (colonne : {continent_col})…")
    cont = df[continent_col].value_counts()
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        cont.plot(kind="bar", ax=ax, color="#EE7733", edgecolor="none")
        ax.set_xlabel("")
        ax.set_ylabel("Nombre d'images")
        ax.set_title("Distribution par continent", fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "02_continent_distribution.png")
        plt.close(fig)

# ──────────────────────────────────────────────────────────────
# 7. VALEURS MANQUANTES
# ──────────────────────────────────────────────────────────────
print("Plot valeurs manquantes…")
miss_pct = missing[missing > 0]
if len(miss_pct):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, max(3, len(miss_pct) * 0.4 + 1)))
        miss_pct.sort_values().plot(kind="barh", ax=ax, color="#CC3311")
        ax.set_xlabel("% manquant")
        ax.set_title("Valeurs manquantes par colonne", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "08_missing_values.png")
        plt.close(fig)

# ──────────────────────────────────────────────────────────────
# 8. CELLULES K-MEANS  (si pkl disponible)
# ──────────────────────────────────────────────────────────────
if CELLS_PKL.exists():
    print("Analyse des cellules k-means…")
    with open(CELLS_PKL, "rb") as f:
        cells = pickle.load(f)

    id_to_cell  = cells["id_to_cell"]
    centers     = cells["centers"]
    n_cells     = cells["n_cells"]

    # Nb images par cellule
    cell_counts = Counter(id_to_cell.values())
    counts_arr  = np.array([cell_counts.get(i, 0) for i in range(n_cells)])

    log(f"\n── Cellules k-means ({n_cells}) ──────────────────────────────────")
    log(f"  Images assignées    : {sum(counts_arr):,}")
    log(f"  Cellules vides      : {(counts_arr == 0).sum()}")
    log(f"  Min / Max / Moyenne : {counts_arr.min()} / {counts_arr.max()} / {counts_arr.mean():.1f}")

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Distribution du nb d'images par cellule
        axes[0].hist(counts_arr[counts_arr > 0], bins=60, color="#228833", edgecolor="none")
        axes[0].set_xlabel("Images par cellule")
        axes[0].set_ylabel("Nombre de cellules")
        axes[0].set_title("Distribution taille des cellules", fontweight="bold")

        # Centroïdes sur une projection équirectangulaire
        axes[1].scatter(centers[:, 1], centers[:, 0],
                        c=counts_arr, cmap="plasma", s=8, alpha=0.7)
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        axes[1].set_title("Centroïdes k-means colorés par nb d'images", fontweight="bold")
        sm = plt.cm.ScalarMappable(cmap="plasma",
                                   norm=plt.Normalize(counts_arr.min(), counts_arr.max()))
        plt.colorbar(sm, ax=axes[1], label="Images")

        fig.tight_layout()
        fig.savefig(OUT_DIR / "05_cell_distribution.png")
        plt.close(fig)

    # Dispersion intra-cellule (Haversine moyenne entre chaque point et son centroïde)
    print("Calcul dispersion intra-cellule (sample 50k)…")

    df_sub = df[["id", "latitude", "longitude"]].dropna().copy()
    df_sub["id"] = df_sub["id"].astype(str)
    df_sub = df_sub[df_sub["id"].isin(id_to_cell)].sample(
        n=min(50_000, len(df_sub)), random_state=42
    )
    df_sub["cell"] = df_sub["id"].map(id_to_cell)
    df_sub["clat"] = df_sub["cell"].map(lambda c: centers[c, 0])
    df_sub["clon"] = df_sub["cell"].map(lambda c: centers[c, 1])

    def haversine_vec(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    df_sub["dist_km"] = haversine_vec(
        df_sub["latitude"].values,  df_sub["longitude"].values,
        df_sub["clat"].values,      df_sub["clon"].values,
    )

    log(f"\n── Dispersion intra-cellule (sample 50k) ──────────────────")
    log(f"  Médiane : {df_sub['dist_km'].median():.1f} km")
    log(f"  Moyenne : {df_sub['dist_km'].mean():.1f} km")
    log(f"  P90     : {df_sub['dist_km'].quantile(0.90):.1f} km")
    log(f"  P95     : {df_sub['dist_km'].quantile(0.95):.1f} km")
    log(f"  Max     : {df_sub['dist_km'].max():.1f} km")

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_sub["dist_km"], bins=100, color="#AA3377", edgecolor="none")
        ax.axvline(df_sub["dist_km"].median(), color="black", linestyle="--",
                   label=f"Médiane {df_sub['dist_km'].median():.0f} km")
        ax.set_xlabel("Distance au centroïde (km)")
        ax.set_ylabel("Nombre d'images")
        ax.set_title("Dispersion intra-cellule (Haversine)", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "10_haversine_intra_cell.png")
        plt.close(fig)

else:
    log(f"\n[INFO] Fichier pkl introuvable : {CELLS_PKL} — section cellules ignorée")

# ──────────────────────────────────────────────────────────────
# 9. RÉSOLUTION DES IMAGES  (si colonnes width/height présentes)
# ──────────────────────────────────────────────────────────────
w_col = next((c for c in df.columns if c.lower() in ["width", "w", "img_width"]), None)
h_col = next((c for c in df.columns if c.lower() in ["height", "h", "img_height"]), None)
if w_col and h_col:
    print("Distribution résolutions…")
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, col in zip(axes, [w_col, h_col]):
            df[col].dropna().astype(float).hist(bins=60, ax=ax, color="#66CCEE")
            ax.set_xlabel(col)
            ax.set_title(f"Distribution {col}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "07_image_resolution.png")
        plt.close(fig)

# ──────────────────────────────────────────────────────────────
# 10. SAUVEGARDE DU RAPPORT TEXTE
# ──────────────────────────────────────────────────────────────
log(f"\n{'='*60}")
log("Tous les graphiques sont dans : eda_outputs_test/")
log("="*60)

with open(OUT_DIR / "eda_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\nEDA terminée. Rapport : {OUT_DIR / 'eda_report.txt'}")
print(f"Graphiques       : {OUT_DIR}/")