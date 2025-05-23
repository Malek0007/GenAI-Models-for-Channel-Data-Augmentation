import json
import numpy as np
import folium

def interpolate_points_on_line(coords, total_points):
    if len(coords) < 2:
        return []

    distances = [np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1])) for i in range(len(coords) - 1)]
    total_distance = sum(distances)

    if total_distance == 0:
        return []

    points_per_segment = [int((d / total_distance) * total_points) for d in distances]
    diff = total_points - sum(points_per_segment)
    for i in range(abs(diff)):
        points_per_segment[i % len(points_per_segment)] += np.sign(diff)

    new_points = []
    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i + 1]

        lats = np.linspace(lat1, lat2, points_per_segment[i] + 1, endpoint=False)[1:]
        lons = np.linspace(lon1, lon2, points_per_segment[i] + 1, endpoint=False)[1:]

        new_points.extend(zip(lats, lons))

    return new_points[:total_points]

def find_closest_index(coord_list, target_coord):
    distances = [np.linalg.norm(np.array(c) - np.array(target_coord)) for c in coord_list]
    return int(np.argmin(distances))

# Étape 1 : Charger les trajets complets (Passage_5B)
with open("cfmimo/Passage_5B.json", "r") as f:
    passage_data = json.load(f)

coords_all = [(point["latitude"], point["longitude"]) for point in passage_data]

# Étape 2 : Charger les 10 points de départ
with open("10_departure_corrdinates.json", "r") as f:
    departures = json.load(f)

# Étape 3 : Générer 10 segments interpolés à partir des points de départ les plus proches
result = []
for departure in departures:
    departure_coord = (departure["latitude"], departure["longitude"])
    closest_idx = find_closest_index(coords_all, departure_coord)

    # Inclure le point de départ exact avant le reste du trajet
    segment_coords = [departure_coord] + coords_all[closest_idx:]

    interpolated = interpolate_points_on_line(segment_coords, 100)
    result.append([{"latitude": lat, "longitude": lon} for lat, lon in interpolated])

# Étape 4 : Sauvegarder dans un fichier JSON
with open("interpolated3_passage_5B.json", "w") as f:
    json.dump(result, f, indent=4)

# Étape 5 : Visualisation avec Folium
with open("interpolated3_passage_5B.json", "r") as f:
    result = json.load(f)

# Centrer la carte sur le premier point
map_center = result[0][0]['latitude'], result[0][0]['longitude']
m = folium.Map(location=map_center, zoom_start=50)

# Définir des couleurs
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'darkgreen', 'cadetblue', 'pink']

# Ajouter les trajets sous forme de points
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

    # Marqueur de départ
    folium.Marker(
        location=(traj[0]['latitude'], traj[0]['longitude']),
        popup=f"Départ {i+1}",
        icon=folium.Icon(color=color, icon="play", prefix="fa")
    ).add_to(m)

# Sauvegarder dans un fichier HTML
m.save("visualisation_points_passage_5B.html")
