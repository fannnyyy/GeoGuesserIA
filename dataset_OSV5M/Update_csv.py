import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import zipfile

DATA_DIR = Path("/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m")
IMAGES_DIR = DATA_DIR / "images"
METADATA_DIR = DATA_DIR / "metadata"


def find_downloaded_files(base_dir):
    """Trouve tous les fichiers téléchargés (zip et images extraites)"""
    stats = {
        'train': {'zips': [], 'images': set()},
        'test': {'zips': [], 'images': set()}
    }
    
    for split in ['train', 'test']:
        split_dir = base_dir / split
        if not split_dir.exists():
            print(f"⚠️  Dossier {split} n'existe pas")
            continue
        
        # Chercher les zips
        zips = list(split_dir.glob("*.zip"))
        stats[split]['zips'] = [z.name for z in zips]
        
        # Chercher les images extraites (dans des sous-dossiers)
        for subdir in split_dir.iterdir():
            if subdir.is_dir():
                images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                for img in images:
                    # Extraire l'ID de l'image (nom du fichier sans extension)
                    img_id = img.stem
                    stats[split]['images'].add(img_id)
        
        print(f"\nSplit: {split.upper()}")
        print(f"  Fichiers ZIP: {len(stats[split]['zips'])}")
        print(f"  Images extraites: {len(stats[split]['images'])}")
        if stats[split]['zips']:
            print(f"  ZIPs trouvés: {', '.join(sorted(stats[split]['zips'])[:10])}")
            if len(stats[split]['zips']) > 10:
                print(f"              ... et {len(stats[split]['zips']) - 10} autres")
    
    return stats

stats = find_downloaded_files(IMAGES_DIR)


def extract_remaining_zips(base_dir, stats):
    """Extrait les ZIPs qui n'ont pas encore été extraits"""
    total_extracted = 0
    
    for split in ['train', 'test']:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
        
        zips = list(split_dir.glob("*.zip"))
        if not zips:
            print(f"Aucun ZIP à extraire pour {split}")
            continue
        
        print(f"\nExtraction de {len(zips)} ZIPs pour {split}...")
        
        for zip_file in tqdm(zips, desc=f"Extraction {split}"):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Extraire dans le même dossier
                    zip_ref.extractall(split_dir)
                
                # Supprimer le ZIP après extraction réussie
                zip_file.unlink()
                total_extracted += 1
                
            except Exception as e:
                print(f"Erreur avec {zip_file.name}: {e}")
    
    print(f"\nTotal de ZIPs extraits: {total_extracted}")
    return total_extracted

# Extraire les ZIPs
extracted = extract_remaining_zips(IMAGES_DIR, stats)

# Re-scanner après extraction
if extracted > 0:
    print("\nRe-scan après extraction...")
    stats = find_downloaded_files(IMAGES_DIR)


def filter_csv(csv_path, available_image_ids, split_name):
    """Filtre un CSV pour ne garder que les images disponibles"""
    
    if not csv_path.exists():
        print(f"Fichier CSV non trouvé: {csv_path}")
        return None
    
    print(f"\nTraitement de {csv_path.name}...")
    
    # Charger le CSV
    df = pd.read_csv(csv_path)
    print(f"  Total lignes: {len(df):,}")
    
    # Déterminer quelle colonne contient l'ID de l'image
    # Les colonnes possibles sont: 'id', 'image_id', 'filename', etc.
    id_column = None
    for col in ['id', 'image_id', 'filename', 'key']:
        if col in df.columns:
            id_column = col
            break
    
    if id_column is None:
        print(f"Impossible de trouver la colonne ID. Colonnes disponibles: {df.columns.tolist()}")
        # Essayer avec l'index
        print("  Tentative d'utiliser l'index comme ID...")
        df['temp_id'] = df.index.astype(str)
        id_column = 'temp_id'
    
    # Filtrer
    before_count = len(df)
    
    # Extraire juste le nom du fichier sans extension si nécessaire
    df['clean_id'] = df[id_column].astype(str).apply(lambda x: Path(x).stem)
    
    df_filtered = df[df['clean_id'].isin(available_image_ids)].copy()
    df_filtered = df_filtered.drop(columns=['clean_id'])
    
    after_count = len(df_filtered)
    kept_percent = (after_count / before_count * 100) if before_count > 0 else 0
    
    print(f"  Lignes gardées: {after_count:,} / {before_count:,} ({kept_percent:.1f}%)")
    
    return df_filtered

# Filtrer les CSV
filtered_dfs = {}

for split in ['train', 'test']:
    csv_path = METADATA_DIR / f"{split}.csv"
    available_ids = stats[split]['images']
    
    if len(available_ids) > 0:
        df_filtered = filter_csv(csv_path, available_ids, split)
        if df_filtered is not None:
            filtered_dfs[split] = df_filtered
    else:
        print(f"\nAucune image disponible pour {split}, skip du CSV")



output_dir = DATA_DIR / "metadata_filtered"
output_dir.mkdir(exist_ok=True)

for split, df in filtered_dfs.items():
    output_path = output_dir / f"{split}_filtered.csv"
    df.to_csv(output_path, index=False)
    print(f"Sauvegardé: {output_path}")
    print(f"   {len(df):,} entrées")



total_images = sum(len(s['images']) for s in stats.values())
print(f"\nImages disponibles:")
print(f"  Train: {len(stats['train']['images']):,}")
print(f"  Test: {len(stats['test']['images']):,}")
print(f"  TOTAL: {total_images:,}")

if filtered_dfs:
    print(f"\nCSV filtrés créés:")
    for split, df in filtered_dfs.items():
        print(f"  {split}: {len(df):,} entrées → data/metadata_filtered/{split}_filtered.csv")


summary_path = DATA_DIR / "dataset_summary.txt"
with open(summary_path, 'w') as f:
    f.write("OSV5M Dataset - Résumé du téléchargement partiel\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Images disponibles:\n")
    f.write(f"  Train: {len(stats['train']['images']):,}\n")
    f.write(f"  Test: {len(stats['test']['images']):,}\n")
    f.write(f"  TOTAL: {total_images:,}\n\n")
    
    f.write(f"CSV filtrés:\n")
    for split, df in filtered_dfs.items():
        f.write(f"  {split}: {len(df):,} entrées\n")
    
    f.write(f"\nUtilisation:\n")
    f.write(f"  - Utiliser les CSV dans: data/metadata_filtered/\n")
    f.write(f"  - Images dans: data/images/{'{train,test}'}/\n")

