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

def analyze_dataset_structure(json_metadata_path):
    """
    Analyse la structure du dataset et extrait les métadonnées à partir du JSON
    """
    import json
    from pathlib import Path
    from collections import Counter
        
    # Charger le fichier JSON de métadonnées
    with open(json_metadata_path, 'r', encoding='utf-8') as f:
        json_metadata = json.load(f)
    
    country_counts = Counter()
    
    # Parcourir les métadonnées JSON
    for filename, info in json_metadata.items():
        country_name = info.get('city')  # Dans votre JSON, le pays est dans le champ "city"
        
        if country_name:
            # Incrémenter le compteur pour ce pays
            country_counts[country_name] += 1
    
    return country_counts

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



def create_world_map(location_counts, output_file='world_distribution_map_kaggle.html'):
    """
    Crée une carte mondiale interactive avec des markers pour chaque ville
    """    
    # Coordonnées GPS des villes et pays
    location_coordinates = {
        # Villes américaines
        'California': (36.7783, -119.4179),
        'NYC': (40.7128, -74.0060),
        'nyc': (40.7128, -74.0060),
        'SanFrancisco': (37.7749, -122.4194),
        'Chicago': (41.8781, -87.6298),
        'Seattle': (47.6062, -122.3321),
        'Los Angeles': (34.0522, -118.2437),
        'Boston': (42.3601, -71.0589),
        'Miami': (25.7617, -80.1918),
        'Washington': (38.9072, -77.0369),
        'Las Vegas': (36.1699, -115.1398),
        'Portland': (45.5152, -122.6784),
        'Austin': (30.2672, -97.7431),
        'Denver': (39.7392, -104.9903),
        
        # Villes UK
        'London': (51.5074, -0.1278),
        'scotland': (56.4907, -4.2026),
        'Scotland': (56.4907, -4.2026),
        'england': (52.3555, -1.1743),
        'England': (52.3555, -1.1743),
        'Manchester': (53.4808, -2.2426),
        'Edinburgh': (55.9533, -3.1883),
        'Liverpool': (53.4084, -2.9916),
        'Birmingham': (52.4862, -1.8904),
        'Glasgow': (55.8642, -4.2518),
        
        # Villes françaises
        'Paris': (48.8566, 2.3522),
        'Lyon': (45.7640, 4.8357),
        'Marseille': (43.2965, 5.3698),
        'Toulouse': (43.6047, 1.4442),
        'Nice': (43.7102, 7.2620),
        'Bordeaux': (44.8378, -0.5792),
        'Strasbourg': (48.5734, 7.7521),
        
        # Villes allemandes
        'Berlin': (52.5200, 13.4050),
        'Munich': (48.1351, 11.5820),
        'Hamburg': (53.5511, 9.9937),
        'Frankfurt': (50.1109, 8.6821),
        'Cologne': (50.9375, 6.9603),
        
        # Villes espagnoles
        'Barcelona': (41.3851, 2.1734),
        'Madrid': (40.4168, -3.7038),
        'Valencia': (39.4699, -0.3763),
        'Seville': (37.3891, -5.9845),
        
        # Villes italiennes
        'Rome': (41.9028, 12.4964),
        'Milan': (45.4642, 9.1900),
        'Venice': (45.4408, 12.3155),
        'Florence': (43.7696, 11.2558),
        'Naples': (40.8518, 14.2681),
        
        # Villes japonaises
        'Tokyo': (35.6762, 139.6503),
        'Osaka': (34.6937, 135.5023),
        'Kyoto': (35.0116, 135.7681),
        
        # Villes chinoises
        'Beijing': (39.9042, 116.4074),
        'Shanghai': (31.2304, 121.4737),
        'Hong Kong': (22.3193, 114.1694),
        
        # Villes canadiennes
        'Toronto': (43.6532, -79.3832),
        'Vancouver': (49.2827, -123.1207),
        'Montreal': (45.5017, -73.5673),
        
        # Villes australiennes
        'Sydney': (33.8688, 151.2093),
        'Melbourne': (37.8136, 144.9631),
        'Brisbane': (27.4698, 153.0251),
        
        # Autres villes importantes
        'Amsterdam': (52.3676, 4.9041),
        'Brussels': (50.8503, 4.3517),
        'Vienna': (48.2082, 16.3738),
        'Stockholm': (59.3293, 18.0686),
        'Copenhagen': (55.6761, 12.5683),
        'Oslo': (59.9139, 10.7522),
        'Prague': (50.0755, 14.4378),
        'Budapest': (47.4979, 19.0402),
        'Athens': (37.9838, 23.7275),
        'Lisbon': (38.7223, -9.1393),
        'Dublin': (53.3498, -6.2603),
        'Moscow': (55.7558, 37.6173),
        'Istanbul': (41.0082, 28.9784),
        'Dubai': (25.2048, 55.2708),
        'Singapore': (1.3521, 103.8198),
        'Bangkok': (13.7563, 100.5018),
        'Seoul': (37.5665, 126.9780),
        'Mumbai': (19.0760, 72.8777),
        'Delhi': (28.7041, 77.1025),
        'Mexico City': (19.4326, -99.1332),
        'Buenos Aires': (34.6037, -58.3816),
        'Rio de Janeiro': (22.9068, -43.1729),
        'Sao Paulo': (23.5505, -46.6333),
        'Cairo': (30.0444, 31.2357),
        'Cape Town': (33.9249, 18.4241),
        
        # Pays (coordonnées centrales)
        'France': (46.2276, 2.2137),
        'Germany': (51.1657, 10.4515),
        'Spain': (40.4637, -3.7492),
        'Italy': (41.8719, 12.5674),
        'Canada': (56.1304, -106.3468),
        'Australia': (25.2744, 133.7751),
        'Japan': (36.2048, 138.2529),
        'China': (35.8617, 104.1954),
        'USA': (37.0902, -95.7129),
        'United States': (37.0902, -95.7129),
        'UK': (55.3781, -3.4360),
        'United Kingdom': (55.3781, -3.4360),
        'India': (20.5937, 78.9629),
        'Brazil': (14.2350, -51.9253),
        'Russia': (61.5240, 105.3188),
        'Mexico': (23.6345, -102.5528),
        'Argentina': (38.4161, -63.6167),
        'South Africa': (30.5595, 22.9375),
        'Turkey': (38.9637, 35.2433),
        'Thailand': (15.8700, 100.9925),
        'South Korea': (35.9078, 127.7669),
        'Netherlands': (52.1326, 5.2913),
        'Belgium': (50.5039, 4.4699),
        'Switzerland': (46.8182, 8.2275),
        'Sweden': (60.1282, 18.6435),
        'Norway': (60.4720, 8.4689),
        'Poland': (51.9194, 19.1451),
        'Austria': (47.5162, 14.5501),
        'Greece': (39.0742, 21.8243),
        'Portugal': (39.3999, -8.2245),
        'Egypt': (26.8206, 30.8025),
        'Afghanistan': (33.9391, 67.7100),
        'Indonesia': (0.7893, 113.9213),
        'Philippines': (12.8797, 121.7740),
        'Vietnam': (14.0583, 108.2772),
        'Malaysia': (4.2105, 101.9758),
        'New Zealand': (40.9006, 174.8860),
        'Chile': (35.6751, -71.5430),
        'Colombia': (4.5709, -74.2973),
        'Peru': (9.1900, -75.0152),
        'Venezuela': (6.4238, -66.5897),
        'Morocco': (31.7917, -7.0926),
        'Kenya': (0.0236, 37.9062),
        'Nigeria': (9.0820, 8.6753),
        'Israel': (31.0461, 34.8516),
        'Saudi Arabia': (23.8859, 45.0792),
        'UAE': (23.4241, 53.8478),
        'Pakistan': (30.3753, 69.3451),
        'Bangladesh': (23.6850, 90.3563),
        'Sri Lanka': (7.8731, 80.7718),
    }
    
    # Préparer les données
    lats, lons, names, counts, sizes = [], [], [], [], []
    
    for location, count in location_counts.items():
        if location in location_coordinates:
            lat, lon = location_coordinates[location]
            lats.append(lat)
            lons.append(lon)
            names.append(location)
            counts.append(count)
            # Taille proportionnelle au nombre d'images
            sizes.append(max(5, min(50, count * 0.5)))  # Entre 5 et 50 pixels
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'location': names,
        'lat': lats,
        'lon': lons,
        'images': counts,
        'size': sizes
    })
    
    # Créer la carte avec des markers
    fig = px.scatter_geo(
        df,
        lat='lat',
        lon='lon',
        size='size',
        color='images',
        hover_name='location',
        hover_data={'lat': False, 'lon': False, 'images': ':,', 'size': False},
        color_continuous_scale='YlOrRd',
        labels={'images': 'Nombre d\'images'},
        title='Distribution géographique du dataset',
        projection='natural earth'
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            showcountries=True,
            countrycolor='lightgray',
        ),
        height=600,
        title_x=0.5
    )
    
    fig.write_html(output_file)

def main(dataset_path):
    """
    Fonction principale
    """    
    # Analyser la structure
    country_counts = analyze_dataset_structure(dataset_path)

    
    # Créer les visualisations
    create_distribution_charts(country_counts)
    create_world_map(country_counts)
    

if __name__ == "__main__":
    # Remplacez par le chemin vers votre dataset
    DATASET_PATH = "C:/Users/fanny/OneDrive/Bureau/Cours_CS/GeoGuesserIA/GeoGuesserIA/dataset_2k_random_test/label_association/labels.json"
    
    # Si vous avez une structure différente, modifiez ici:
    # DATASET_PATH = "./mon_dataset"
    
    main(DATASET_PATH)