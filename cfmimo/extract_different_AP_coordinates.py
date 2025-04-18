import json
import numpy as np

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

# Étape 1 : Charger les données JSON
with open("cfmimo\Passage_5B.json", "r") as f:
    data = json.load(f)

# Étape 2 : Extraire les latitudes et longitudes
coords_all = [(point["latitude"], point["longitude"]) for point in data]

# Étape 3 : Générer 10 segments interpolés
result = []
for i in range(10):
    segment_coords = coords_all[i:]
    interpolated = interpolate_points_on_line(segment_coords, 100)
    result.append([{"latitude": lat, "longitude": lon} for lat, lon in interpolated])

# Étape 4 : Sauvegarder dans un fichier JSON
with open("interpolated_passage_5B.json", "w") as f:
    json.dump(result, f, indent=4)
