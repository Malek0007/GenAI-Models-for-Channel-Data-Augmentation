import json
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Charger les points d'accès (AP) et les points jaunes à partir des fichiers JSON
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Charger les données
ap_file_path = "AP_coordinates.json"
accesspoint_file_path = "coordinates_of_100_AP.json"

access_points = load_json(ap_file_path)  # Points d'accès (AP)
accesspoint = load_json(accesspoint_file_path)  # Points jaunes

# Fonction pour calculer la distance Haversine entre deux points géographiques
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Rayon de la Terre en km

    # Convertir les coordonnées en radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Calculer les différences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Formule Haversine
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance en kilomètres
    distance = R * c * 1000
    return distance

# Calculer la matrice de distances entre chaque point d'accès et chaque point jaune
distance_matrix = []

for ap in access_points:
    ap_lat = ap["latitude"]
    ap_lon = ap["longitude"]
    distances = []
    for access_point in accesspoint:
        access_lat = access_point["latitude"]
        access_lon = access_point["longitude"]
        
        # Calculer la distance entre l'AP et le point jaune
        distance = haversine(ap_lat, ap_lon, access_lat, access_lon)
        distances.append(distance)
    distance_matrix.append(distances)

# Sauvegarder la matrice de distances dans un fichier JSON
output_file = "distance_matrix.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(distance_matrix, f, indent=4)

print(f"Matrice de distances sauvegardée dans {output_file}")
