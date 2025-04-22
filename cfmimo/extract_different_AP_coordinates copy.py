import json
import numpy as np
import folium
import webbrowser
import os

def interpolate_points_on_line(coords, total_points):
    if len(coords) < 2:
        return []

    # Calculate distances between consecutive points
    distances = [np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1])) for i in range(len(coords) - 1)]
    total_distance = sum(distances)

    if total_distance == 0:
        return []

    # Calculate how many points to interpolate per segment
    points_per_segment = [int((d / total_distance) * total_points) for d in distances]

    # Adjust so that total number of points matches exactly
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

# Load original coordinates
with open("cfmimo/Passage_5B.json", "r") as f:
    passage_data = json.load(f)

coords_all = [(point["latitude"], point["longitude"]) for point in passage_data]

# Parameters
nb_trajets = 1
total_points = 100

result = []

# First path
trajet_initial = interpolate_points_on_line(coords_all, total_points)
result.append([{"latitude": lat, "longitude": lon} for lat, lon in trajet_initial])

# Subsequent paths
for i in range(1, nb_trajets):
    prev_coords = [(point["latitude"], point["longitude"]) for point in result[i - 1]]
    start_coords = prev_coords[1:]
    new_trajet = interpolate_points_on_line(start_coords, total_points)
    result.append([{"latitude": lat, "longitude": lon} for lat, lon in new_trajet])

# Save output to JSON
with open("interpolated_linked_departures_passage_5B.json", "w") as f:
    json.dump(result, f, indent=4)

# Create Folium map
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

# Automatically open in web browser
map_filename = "visualisation_linked_departures1.html"
m.save(map_filename)
webbrowser.open('file://' + os.path.realpath(map_filename))
