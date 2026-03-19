"""
Pré-calcul des cellules k-means sur les coordonnées GPS du dataset OSV5M.

Ce script doit être lancé UNE SEULE FOIS avant l'entraînement.
Il génère un fichier cells_kmeans.pkl contenant :
    - les centres des cellules (lat, lon)
    - le mapping id_image → idx_cellule

Usage :
    python compute_cells.py
"""

import os
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

N_CELLS   = 1000
CSV_FILE  = os.path.expanduser("~/datasets/OSV5M/test_filtered.csv")
OUT_PATH  = os.path.expanduser("~/datasets/OSV5M/cells_kmeans.pkl")
SEED      = 42

# ─────────────────────────────────────────────
# Chargement des coordonnées
# ─────────────────────────────────────────────

print("Chargement du CSV...")
data = pd.read_csv(CSV_FILE)
data = data.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
print(f"Lignes chargées : {len(data)}")

# Coordonnées en radians sur le cercle unité (même espace que le modèle)
# On utilise lat/lon bruts pour le k-means — plus intuitif pour les centroïdes
coords = data[["latitude", "longitude"]].values.astype(np.float32)

# ─────────────────────────────────────────────
# K-means
# ─────────────────────────────────────────────

print(f"\nCalcul k-means — {N_CELLS} cellules sur {len(coords)} points...")
kmeans = MiniBatchKMeans(
    n_clusters  = N_CELLS,
    random_state= SEED,
    batch_size  = 10_000,
    n_init      = 10,
    max_iter    = 300,
    verbose     = 1,
)
kmeans.fit(coords)

labels   = kmeans.labels_                  # (N,) — indice cellule pour chaque image
centers  = kmeans.cluster_centers_         # (N_CELLS, 2) — [lat, lon] de chaque centroïde

print(f"\nK-means terminé.")
print(f"Inertie finale : {kmeans.inertia_:.2f}")

# ─────────────────────────────────────────────
# Statistiques
# ─────────────────────────────────────────────

counts = np.bincount(labels, minlength=N_CELLS)
print(f"\nImages par cellule — min: {counts.min()}  max: {counts.max()}  "
      f"moyenne: {counts.mean():.1f}  médiane: {np.median(counts):.1f}")

empty = (counts == 0).sum()
if empty > 0:
    print(f"  [WARN] {empty} cellules vides — augmente le dataset ou réduis N_CELLS")

# ─────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────

# Ajoute la colonne cell_idx au dataframe pour vérification
data["cell_idx"] = labels

payload = {
    "n_cells"       : N_CELLS,
    "centers"       : centers,       # (N_CELLS, 2) float32 [lat, lon]
    "kmeans"        : kmeans,        # objet sklearn pour prédire sur nouvelles images
    "id_to_cell"    : dict(zip(data["id"].astype(str), labels)),
    "counts"        : counts,
}

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"\nCellules sauvegardées : {OUT_PATH}")
print("Lance maintenant : python resnet_classification.py")
