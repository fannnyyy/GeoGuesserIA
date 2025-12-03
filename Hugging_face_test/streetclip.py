from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
import json
from pathlib import Path

class StreetCLIPGeolocator:
    def __init__(self):
        """Initialise le modèle StreetCLIP"""
        
        self.model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
        self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
        
        # Utiliser GPU si disponible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ Modèle chargé sur {self.device}")
    
    def predict_location(self, image, choices, top_k=5):
        """
        Prédit la localisation d'une image parmi plusieurs choix
        
        Args:
            image: PIL Image ou chemin vers l'image
            choices: Liste de localisations possibles (villes, pays, régions)
            top_k: Nombre de prédictions à retourner
            
        Returns:
            Liste de tuples (location, probabilité)
        """
        # Charger l'image si c'est un chemin
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Préparer les inputs
        inputs = self.processor(
            text=choices,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Déplacer sur le bon device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
        
        # Trier par probabilité décroissante
        top_indices = torch.argsort(probs, descending=True)[:top_k]
        
        results = [
            (choices[idx], probs[idx].item())
            for idx in top_indices
        ]
        
        return results
    

def example_1_basic_usage():
   
    # Initialiser
    geolocator = StreetCLIPGeolocator()
    
    # Télécharger une image test
    url = "https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Prédire parmi plusieurs villes
    cities = ["San Jose", "San Diego", "Los Angeles", "Las Vegas", "San Francisco"]
    results = geolocator.predict_location(image, cities)
    
    for city, prob in results:
        print(f"  {city:20s} {prob*100:6.2f}%")


if __name__ == "__main__":

    example_1_basic_usage()

    