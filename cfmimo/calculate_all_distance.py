import json
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Charger les points d'accès (AP) et les trajets interpolés
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Haversine: distance en mètres entre 2 points géographiques
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c * 1000  # mètres

# Fichiers d'entrée
ap_file_path = "cfmimo\AP_coordinates.json"
interpolated_file_path = "interpolated_linked_departures_passage_5B.json"

# Charger les données
access_points = load_json(ap_file_path)               # 8 AP
all_trajets = load_json(interpolated_file_path)       # 10 trajets × 100 points

# Liste pour stocker toutes les matrices Dlk (Dlk1, Dlk2, ..., Dlk10)
all_distance_matrices = {}

# Pour chaque trajet (1 à 10)
for trajet_index, trajet_points in enumerate(all_trajets):
    matrix = []  # 8 lignes × 100 colonnes
    
    # Pour chaque AP (8)
    for ap in access_points:
        ap_lat = ap["latitude"]
        ap_lon = ap["longitude"]
        distances = []

        # Pour chaque point jaune dans ce trajet (100)
        for point in trajet_points:
            point_lat = point["latitude"]
            point_lon = point["longitude"]
            distance = haversine(ap_lat, ap_lon, point_lat, point_lon)
            distances.append(distance)
        
        matrix.append(distances)

    # Stocker la matrice sous forme Dlk1, Dlk2, ..., Dlk10
    all_distance_matrices[f"Dlk{trajet_index + 1}"] = matrix

# Sauvegarder toutes les matrices dans un seul fichier JSON
output_file = "distance_matrices_all_Dlk.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_distance_matrices, f, indent=4)

print(f"Toutes les matrices Dlk sauvegardées dans {output_file}")
