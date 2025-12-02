import os
import csv
import json
from pathlib import Path

def extract_metadata_from_filename(filename):
    """
    Extrait les métadonnées depuis un nom de fichier Im2GPS
    Format: Ville_Index_PhotoID_Hash_ServerID_UserID.jpg
    Exemple: Abidjan_00001_999026917_1838ecb7ef_1158_10371983@N06.jpg
    """
    # Enlever l'extension
    name_without_ext = filename.rsplit('.', 1)[0]
    
    # Séparer les parties
    parts = name_without_ext.split('_')
    
    if len(parts) >= 6:
        city = parts[0]
        index = parts[1]
        flickr_photo_id = parts[2]
        hash_code = parts[3]
        server_id = parts[4]
        user_id = parts[5]
        
        return {
            'filename': filename,
            'city': city,
            'index': index,
            'flickr_photo_id': flickr_photo_id,
            'hash_code': hash_code,
            'server_id': server_id,
            'flickr_user_id': user_id
        }
    else:
        print(f"Format inattendu pour: {filename}")
        return None

def process_image_folder(folder_path, output_csv='labels.csv', output_json='labels.json'):
    """
    Parcourt un dossier d'images et extrait les métadonnées
    """
    folder = Path(folder_path)
    
    # Extensions d'images à chercher
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    
    metadata_list = []
    
    # Parcourir tous les fichiers
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            metadata = extract_metadata_from_filename(file_path.name)
            if metadata:
                metadata_list.append(metadata)
    
    print(f"{len(metadata_list)} images traitées")
    
    # Sauvegarder en CSV
    if metadata_list:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'city', 'index', 'flickr_photo_id', 
                         'hash_code', 'server_id', 'flickr_user_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(metadata_list)
        
        print(f"CSV sauvegardé: {output_csv}")
        
        # Sauvegarder en JSON (format dictionnaire avec filename comme clé)
        json_data = {item['filename']: {k: v for k, v in item.items() if k != 'filename'} 
                    for item in metadata_list}
        
        with open(output_json, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f" JSON sauvegardé: {output_json}")
        
        # Afficher un aperçu
        print("\n Aperçu des 5 premières entrées:")
        for item in metadata_list[:5]:
            print(f"  - {item['filename']} → Ville: {item['city']}, Flickr ID: {item['flickr_photo_id']}")
        
        # Statistiques
        cities = {}
        for item in metadata_list:
            city = item['city']
            cities[city] = cities.get(city, 0) + 1
        
        print(f"\n {len(cities)} villes différentes trouvées:")
        for city, count in sorted(cities.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {city}: {count} images")
    
    return metadata_list

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacez par le chemin vers votre dossier d'images
    folder_path = "C:/Users/fanny/OneDrive/Bureau/Cours_CS/GeoGuesserIA/GeoGuesserIA/dataset/2k_random_test"  # Dossier courant, ou mettez le chemin complet
    
    print("Extraction des métadonnées depuis les noms de fichiers...\n")
    
    metadata = process_image_folder(
        folder_path=folder_path,
        output_csv='labels.csv',
        output_json='labels.json'
    )
    
    print("\n Terminé !")