"""
Pré-calcul des cellules k-means sur les coordonnées GPS du dataset OSV5M.

Ce script doit être lancé UNE SEULE FOIS avant l'entraînement.
Il génère un fichier cells_kmeans.pkl contenant :
    - centers      : centroïdes k-means (lat, lon)
    - mean_centers : moyennes GPS réelles des images par cellule (plus précis)
    - le mapping id_image → idx_cellule

Usage :
    python compute_cells.py
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

N_CELLS  = 1000
CSV_FILE = os.path.expanduser("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/train_filtered_v2.csv")
OUT_PATH = os.path.expanduser("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/resnet/resnet_full_classif/cells_kmeans.pkl")
SEED     = 42

# ─────────────────────────────────────────────
# Chargement des coordonnées
# ─────────────────────────────────────────────

print("Chargement du CSV...")
data = pd.read_csv(CSV_FILE)
data = data.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
print(f"Lignes chargées : {len(data)}")

coords = data[["latitude", "longitude"]].values.astype(np.float32)

# ─────────────────────────────────────────────
# K-means
# ─────────────────────────────────────────────

print(f"\nCalcul k-means, {N_CELLS} cellules sur {len(coords)} points...")
kmeans = MiniBatchKMeans(
    n_clusters   = N_CELLS,
    random_state = SEED,
    batch_size   = 10_000,
    n_init       = 10,
    max_iter     = 300,
    verbose      = 1,
)
kmeans.fit(coords)

labels  = kmeans.labels_
centers = kmeans.cluster_centers_

print(f"\nK-means terminé.")
print(f"Inertie finale : {kmeans.inertia_:.2f}")

# ─────────────────────────────────────────────
# Statistiques
# ─────────────────────────────────────────────

data["cell_idx"] = labels
counts = np.bincount(labels, minlength=N_CELLS)
print(f"\nImages par cellule, min: {counts.min()}  max: {counts.max()}  "
      f"moyenne: {counts.mean():.1f}  médiane: {np.median(counts):.1f}")

empty = (counts == 0).sum()
if empty > 0:
    print(f"  [WARN] {empty} cellules vides, augmente le dataset ou réduis N_CELLS")

# ─────────────────────────────────────────────
# Moyenne GPS réelle par cellule
# ─────────────────────────────────────────────
# Plus précis que le centroïde k-means (euclidien) :
# on prend la moyenne des vraies coordonnées GPS des images de chaque cellule.

print("\nCalcul des moyennes GPS réelles par cellule...")
mean_centers = np.zeros((N_CELLS, 2), dtype=np.float32)
for cell_id in range(N_CELLS):
    mask = data["cell_idx"] == cell_id
    if mask.sum() > 0:
        mean_centers[cell_id, 0] = data.loc[mask, "latitude"].mean()
        mean_centers[cell_id, 1] = data.loc[mask, "longitude"].mean()
    else:
        # Cellule vide : fallback sur le centroïde k-means
        mean_centers[cell_id] = centers[cell_id]

# Comparaison entre les deux méthodes
diffs = np.sqrt(
    (mean_centers[:, 0] - centers[:, 0])**2 +
    (mean_centers[:, 1] - centers[:, 1])**2
)
print(f"Écart moyen centroïde vs moyenne GPS : {diffs.mean():.4f}°")
print(f"Écart max                            : {diffs.max():.4f}°")

# ─────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────

payload = {
    "n_cells"      : N_CELLS,
    "centers"      : centers,       # (N_CELLS, 2) centroïdes k-means [lat, lon]
    "mean_centers" : mean_centers,  # (N_CELLS, 2) moyennes GPS réelles [lat, lon]
    "kmeans"       : kmeans,
    "id_to_cell"   : dict(zip(data["id"].astype(str), labels)),
    "counts"       : counts,
}

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"\nCellules sauvegardées : {OUT_PATH}")
print("Lance maintenant : python resnet_classification.py")
