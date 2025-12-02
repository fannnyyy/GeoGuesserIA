import os
import csv
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Pour la carte mondiale
try:
    import plotly.express as px
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Installez plotly pour la carte interactive: pip install plotly pandas")

def analyze_dataset_structure(dataset_path):
    """
    Analyse la structure du dataset et extrait les métadonnées
    """
    dataset_path = Path(dataset_path)
    
    metadata = []
    country_counts = Counter()
    
    
        
    # Parcourir les dossiers de pays
    for country_folder in dataset_path.iterdir():
        if not country_folder.is_dir():
            continue
            
        country_name = country_folder.name
        
        # Compter les images dans ce pays
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
        images = [f for f in country_folder.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions]
        
        country_counts[country_name] += len(images)
        
        # Ajouter aux métadonnées
        for img_path in images:
            metadata.append({
                'filename': img_path.name,
                'country': country_name,
                #'split': split_name,
                'path': str(img_path.relative_to(dataset_path))
            })
            
            
    return metadata, country_counts

def save_metadata_to_csv(metadata, output_file='dataset_metadata_kaggle.csv'):
    """
    Sauvegarde les métadonnées en CSV
    """
    if not metadata:
        print("Aucune métadonnée à sauvegarder")
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'country', 'path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"\n CSV sauvegardé: {output_file}")

def save_metadata_to_json(metadata, country_counts, output_file='dataset_metadata_kaggle.json'):
    """
    Sauvegarde les métadonnées en JSON avec statistiques
    """
    json_data = {
        'statistics': {
            'total_images': len(metadata),
            'total_countries': len(country_counts),
            'images_per_country': dict(country_counts)
        },
        'images': metadata
    }
    
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
    
    print(f" JSON sauvegardé: {output_file}")

def create_distribution_charts(country_counts, output_prefix='distribution'):
    """
    Crée des graphiques de distribution
    """
    if not country_counts:
        print(" Pas de données pour créer les graphiques")
        return
    
    # Trier par nombre d'images (décroissant)
    sorted_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)
    countries, counts = zip(*sorted_countries)
    
    # Figure avec 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Top 20 pays
    top_n = min(20, len(countries))
    ax1.barh(range(top_n), counts[:top_n], color='steelblue')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(countries[:top_n])
    ax1.invert_yaxis()
    ax1.set_xlabel('Nombre d\'images')
    ax1.set_title(f'Top {top_n} des pays les plus représentés')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Distribution générale (histogramme)
    ax2.hist(counts, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Nombre d\'images par pays')
    ax2.set_ylabel('Nombre de pays')
    ax2.set_title('Distribution du nombre d\'images')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_charts_kaggle.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_world_map(country_counts, output_file='world_distribution_map_kaggle.html'):
    """
    Crée une carte mondiale interactive avec Plotly
    """
    if not PLOTLY_AVAILABLE:
        print(" Carte mondiale non disponible (installez plotly)")
        return
    
    if not country_counts:
        print(" Pas de données pour créer la carte")
        return
    
    # Créer un DataFrame
    df = pd.DataFrame([
        {'country': country, 'images': count}
        for country, count in country_counts.items()
    ])
    
    # Mapping des noms de pays vers les codes ISO (approximatif)
    # Plotly utilise les codes ISO-3
    country_mapping = {
        'USA': 'USA', 'United States': 'USA',
        'France': 'FRA', 'Germany': 'DEU', 'Spain': 'ESP',
        'Italy': 'ITA', 'UK': 'GBR', 'United Kingdom': 'GBR',
        'Canada': 'CAN', 'Australia': 'AUS', 'Brazil': 'BRA',
        'Russia': 'RUS', 'China': 'CHN', 'Japan': 'JPN',
        'India': 'IND', 'Mexico': 'MEX', 'Argentina': 'ARG',
        'South Africa': 'ZAF', 'Egypt': 'EGY', 'Turkey': 'TUR',
        'Poland': 'POL', 'Netherlands': 'NLD', 'Belgium': 'BEL',
        'Switzerland': 'CHE', 'Sweden': 'SWE', 'Norway': 'NOR',
        'Denmark': 'DNK', 'Finland': 'FIN', 'Greece': 'GRC',
        'Portugal': 'PRT', 'Austria': 'AUT', 'Czech Republic': 'CZE',
        'Hungary': 'HUN', 'Romania': 'ROU', 'Ukraine': 'UKR',
        'Thailand': 'THA', 'Vietnam': 'VNM', 'Indonesia': 'IDN',
        'Philippines': 'PHL', 'Malaysia': 'MYS', 'Singapore': 'SGP',
        'South Korea': 'KOR', 'New Zealand': 'NZL', 'Chile': 'CHL',
        'Colombia': 'COL', 'Peru': 'PER', 'Venezuela': 'VEN',
        'Morocco': 'MAR', 'Kenya': 'KEN', 'Nigeria': 'NGA',
        'Israel': 'ISR', 'Saudi Arabia': 'SAU', 'UAE': 'ARE',
        'Pakistan': 'PAK', 'Bangladesh': 'BGD', 'Sri Lanka': 'LKA',
        'Ireland': 'IRL', 'Croatia': 'HRV', 'Serbia': 'SRB',
        'Bulgaria': 'BGR', 'Slovakia': 'SVK', 'Slovenia': 'SVN',
        'Lithuania': 'LTU', 'Latvia': 'LVA', 'Estonia': 'EST',
        'Iceland': 'ISL', 'Luxembourg': 'LUX', 'Albania': 'ALB',
        'Armenia': 'ARM', 'Azerbaijan': 'AZE', 'Georgia': 'GEO',
        'Kazakhstan': 'KAZ', 'Tunisia': 'TUN', 'Algeria': 'DZA',
        'Jordan': 'JOR', 'Lebanon': 'LBN', 'Uruguay': 'URY',
        'Paraguay': 'PRY', 'Bolivia': 'BOL', 'Ecuador': 'ECU',
        'Panama': 'PAN', 'Costa Rica': 'CRI', 'Guatemala': 'GTM',
        'Honduras': 'HND', 'Nicaragua': 'NIC', 'El Salvador': 'SLV',
        'Dominican Republic': 'DOM', 'Cuba': 'CUB', 'Jamaica': 'JAM',
        'Mongolia': 'MNG', 'Nepal': 'NPL', 'Cambodia': 'KHM',
        'Laos': 'LAO', 'Myanmar': 'MMR', 'Taiwan': 'TWN',
        'Hong Kong': 'HKG', 'Macau': 'MAC', 'Ethiopia': 'ETH',
        'Ghana': 'GHA', 'Tanzania': 'TZA', 'Uganda': 'UGA',
        'Zimbabwe': 'ZWE', 'Zambia': 'ZMB', 'Botswana': 'BWA',
        'Namibia': 'NAM', 'Mozambique': 'MOZ', 'Madagascar': 'MDG',
        'Senegal': 'SEN', "Cote d'Ivoire": 'CIV', 'Ivory Coast': 'CIV',
        'Cameroon': 'CMR', 'Angola': 'AGO', 'Congo': 'COG',
        'Democratic Republic of Congo': 'COD', 'Mali': 'MLI',
        'Niger': 'NER', 'Chad': 'TCD', 'Sudan': 'SDN',
        'Libya': 'LBY', 'Afghanistan': 'AFG', 'Iraq': 'IRQ',
        'Iran': 'IRN', 'Syria': 'SYR', 'Yemen': 'YEM',
        'Oman': 'OMN', 'Kuwait': 'KWT', 'Bahrain': 'BHR',
        'Qatar': 'QAT', 'Cyprus': 'CYP', 'Malta': 'MLT',
        'Belarus': 'BLR', 'Moldova': 'MDA', 'Bosnia': 'BIH',
        'Bosnia and Herzegovina': 'BIH', 'Montenegro': 'MNE',
        'North Macedonia': 'MKD', 'Kosovo': 'XKX',
    }
    
    # Ajouter les codes ISO
    df['iso_alpha'] = df['country'].map(country_mapping)
    
    # Créer la carte
    fig = px.choropleth(
        df,
        locations='iso_alpha',
        color='images',
        hover_name='country',
        hover_data={'iso_alpha': False, 'images': ':,'},
        color_continuous_scale='YlOrRd',
        labels={'images': 'Nombre d\'images'},
        title='Distribution géographique du dataset'
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=600,
        title_x=0.5
    )
    
    fig.write_html(output_file)
    print(f" Carte mondiale sauvegardée: {output_file}")
    print(f"   Ouvrez ce fichier dans votre navigateur pour voir la carte interactive!")


def main(dataset_path):
    """
    Fonction principale
    """    
    # Analyser la structure
    metadata, country_counts = analyze_dataset_structure(dataset_path)
    
    # Sauvegarder les métadonnées
    save_metadata_to_csv(metadata)
    save_metadata_to_json(metadata, country_counts)
    
    # Créer les visualisations
    create_distribution_charts(country_counts)
    create_world_map(country_counts)
    

if __name__ == "__main__":
    # Remplacez par le chemin vers votre dataset
    DATASET_PATH = "C:/Users/fanny/OneDrive/Bureau/Cours_CS/GeoGuesserIA/GeoGuesserIA/dataset/compressed_dataset"
    
    # Si vous avez une structure différente, modifiez ici:
    # DATASET_PATH = "./mon_dataset"
    
    main(DATASET_PATH)