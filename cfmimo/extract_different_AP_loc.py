import json
import numpy as np
import folium
import webbrowser
import os
from geopy.distance import geodesic

# Nouvelle fonction : interpolation géographique équidistante
def interpolate_geodesic_points(coords, total_points):
    if len(coords) < 2:
        return []

    # Distance cumulée le long du chemin
    distances = [0.0]
    for i in range(1, len(coords)):
        d = geodesic(coords[i - 1], coords[i]).meters
        distances.append(distances[-1] + d)

    total_distance = distances[-1]
    if total_distance == 0:
        return []

    # Créer des distances cibles équidistantes
    target_distances = np.linspace(0, total_distance, total_points)

    result = []
    j = 0
    for td in target_distances:
        while j < len(distances) - 1 and distances[j + 1] < td:
            j += 1
        ratio = (td - distances[j]) / (distances[j + 1] - distances[j]) if distances[j + 1] != distances[j] else 0
        lat1, lon1 = coords[j]
        lat2, lon2 = coords[j + 1]
        lat = lat1 + ratio * (lat2 - lat1)
        lon = lon1 + ratio * (lon2 - lon1)
        result.append((lat, lon))

    return result

# Charger les coordonnées originales
with open("cfmimo/Passage_5B.json", "r") as f:
    passage_data = json.load(f)

coords_all = [(point["latitude"], point["longitude"]) for point in passage_data]

# Paramètres
nb_trajets = 10
total_points = 100
result = []

# Premier trajet
trajet_initial = interpolate_geodesic_points(coords_all, total_points)
result.append([{"latitude": lat, "longitude": lon} for lat, lon in trajet_initial])

# Trajets suivants (liés au précédent)
for i in range(1, nb_trajets):
    prev_coords = [(point["latitude"], point["longitude"]) for point in result[i - 1]]
    start_coords = prev_coords[1:]  # Exclure le premier point pour créer une suite
    new_trajet = interpolate_geodesic_points(start_coords, total_points)
    result.append([{"latitude": lat, "longitude": lon} for lat, lon in new_trajet])

# Enregistrer en JSON
with open("interpolated_linked_departures_passage_5B.json", "w") as f:
    json.dump(result, f, indent=4)

# Création de la carte Folium
map_center = result[0][0]['latitude'], result[0][0]['longitude']
m = folium.Map(location=map_center, zoom_start=19)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'darkgreen', 'cadetblue', 'pink']

for i, traj in enumerate(result):
    color = colors[i % len(colors)]
    for j, point in enumerate(traj):
        folium.CircleMarker(
            location=(point['latitude'], point['longitude']),
            radius=2.5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Trajet {i+1} - Point {j+1}"
        ).add_to(m)

    folium.Marker(
        location=(traj[0]['latitude'], traj[0]['longitude']),
        popup=f"Départ {i+1}",
        icon=folium.Icon(color=color, icon="play", prefix="fa")
    ).add_to(m)

# Afficher automatiquement dans le navigateur
map_filename = "visualisation_linked_departures1.html"
m.save(map_filename)
webbrowser.open('file://' + os.path.realpath(map_filename))
