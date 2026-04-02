import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


DATA_DIR = Path("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m")  
METADATA_TRAIN = DATA_DIR / "metadata_filtered" / "rest_filtered.csv"
METADATA_TEST = DATA_DIR / "metadata_filtered" / "samples_filtered.csv"


try:
    df_train = pd.read_csv(METADATA_TRAIN)
    print(f"Train: {len(df_train):,} images")
except Exception as e:
    print(f"Erreur train: {e}")
    df_train = None

try:
    df_test = pd.read_csv(METADATA_TEST)
    print(f"Test: {len(df_test):,} images")
except Exception as e:
    print(f"Erreur test: {e}")
    df_test = None


if df_train is not None and df_test is not None:
    df = pd.concat([df_train, df_test], ignore_index=True)
    df['split'] = ['train'] * len(df_train) + ['test'] * len(df_test)
elif df_train is not None:
    df = df_train
    df['split'] = 'train'
elif df_test is not None:
    df = df_test
    df['split'] = 'test'
else:
    print("Impossible de charger les données")
    exit(1)

print(f"Total: {len(df):,} images")


print(f"  Latitude min: {df['latitude'].min():.4f}")
print(f"  Latitude max: {df['latitude'].max():.4f}")
print(f"  Longitude min: {df['longitude'].min():.4f}")
print(f"  Longitude max: {df['longitude'].max():.4f}")

if 'country' in df.columns:
    print("\nTop 10 pays avec le plus d'images:")
    top_countries = df['country'].value_counts().head(10)
    for country, count in top_countries.items():
        print(f"  {country}: {count:,} images ({count/len(df)*100:.1f}%)")


sample_size = min(50000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

ax1 = axes[0]
ax1.scatter(
    df_sample['longitude'], 
    df_sample['latitude'],
    s=1,
    alpha=0.3,
    c='blue'
)
ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)
ax1.set_title(f'OSV5M Dataset - Distribution globale ({sample_size:,} images)', 
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-180, 180)
ax1.set_ylim(-90, 90)


ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)


ax2 = axes[1]
heatmap, xedges, yedges = np.histogram2d(
    df_sample['longitude'],
    df_sample['latitude'],
    bins=[180, 90],
    range=[[-180, 180], [-90, 90]]
)

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax2.imshow(
    heatmap.T,
    extent=extent,
    origin='lower',
    aspect='auto',
    cmap='hot',
    interpolation='bilinear'
)
ax2.set_xlabel('Longitude', fontsize=12)
ax2.set_ylabel('Latitude', fontsize=12)
ax2.set_title('Heatmap de densité des images', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax2, label='Nombre d\'images')

plt.tight_layout()


output_path = "osv5m_visualization_filtered.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')

if 'split' in df.columns:
    print(f"Répartition Train/Test:")
    print(f"  Train: {len(df[df['split']=='train']):,} images")
    print(f"  Test: {len(df[df['split']=='test']):,} images")
    print(f"  Ratio: {len(df[df['split']=='train'])/len(df[df['split']=='test']):.2f}:1")
