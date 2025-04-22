import json
import numpy as np
import folium

def interpolate_points_on_line(coords, total_points):
    if len(coords) < 2:
        return []

    # Calculer les distances entre les points consécutifs
    distances = [np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1])) for i in range(len(coords) - 1)]
    total_distance = sum(distances)

    if total_distance == 0:
        return []

    # Nombre de segments
    num_segments = len(coords) - 1
    
    # Diviser les points également sur chaque segment
    points_per_segment = [total_points // num_segments] * num_segments

    # Ajouter les points restants (s'il y en a) de manière égale sur chaque segment
    remaining_points = total_points - sum(points_per_segment)
    for i in range(remaining_points):
        points_per_segment[i % num_segments] += 1

    new_points = []
    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i + 1]

        # Générer des points équidistants sur le segment entre lat1, lon1 et lat2, lon2
        lats = np.linspace(lat1, lat2, points_per_segment[i] + 1)[1:]  # On commence au 2e point
        lons = np.linspace(lon1, lon2, points_per_segment[i] + 1)[1:]

        new_points.extend(zip(lats, lons))

    return new_points[:total_points]

# Charger les coordonnées originales
with open("cfmimo/Passage_5B.json", "r") as f:
    passage_data = json.load(f)

coords_all = [(point["latitude"], point["longitude"]) for point in passage_data]

# Paramètres
nb_trajets = 10
total_points = 100

result = []

# Premier trajet
trajet_initial = interpolate_points_on_line(coords_all, total_points)
result.append([{"latitude": lat, "longitude": lon} for lat, lon in trajet_initial])

# Trajets suivants, chacun commence à partir du point 1 du précédent
for i in range(1, nb_trajets):
    prev_coords = [(point["latitude"], point["longitude"]) for point in result[i - 1]]
    start_coords = prev_coords[1:]  # on commence au 2ème point (index 1)
    new_trajet = interpolate_points_on_line(start_coords, total_points)
    result.append([{"latitude": lat, "longitude": lon} for lat, lon in new_trajet])

# Sauvegarder le fichier
with open("interpolated_linked_departures_passage_5B.json", "w") as f:
    json.dump(result, f, indent=4)

# Visualisation
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

m.save("visualisation_linked_departures3.html")
