import sys
import json
import os
import io
import re
import numpy as np
import csv
from PyQt5 import QtWidgets, QtWebEngineWidgets
import folium
from folium.plugins import Draw

# Fichiers
json_file_path = "cfmimo\\AP_coordinates.json"
passage_5A_file_path = "data\\FST_Passage5A_8APx1.csv"
passage_5B_file_path = "data\\FST_Passage5B_8APx1.csv"

# Charger fichier JSON
def load_json(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except json.JSONDecodeError:
        print(f"Erreur de lecture du fichier {file_path}.")
    return []

# Charger fichier CSV
def load_csv(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                try:
                    data.append({
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "altitude": float(row["altitude (m)"]),
                        "timestamp": row["time"]
                    })
                except (KeyError, ValueError) as e:
                    print(f"Erreur lors de la lecture d'une ligne : {e}")
    else:
        print(f"Le fichier {file_path} n'existe pas.")
    return data

existing_coordinates = load_json(json_file_path)
passage_5A_coordinates = load_csv(passage_5A_file_path)
passage_5B_coordinates = load_csv(passage_5B_file_path)

# Initialiser l'application Qt
app = QtWidgets.QApplication(sys.argv)

# Créer la carte Folium
m = folium.Map(location=[50.61216, 3.14], zoom_start=17)

# Ajouter les points d'accès existants (AP)
for ap in existing_coordinates:
    folium.Marker(
        location=[ap["latitude"], ap["longitude"]],
        popup=ap["name"],
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

# Ajouter le trajet rouge (B → A)
passage_5A_latlngs = [[p["latitude"], p["longitude"]] for p in passage_5A_coordinates]
for passage in passage_5A_coordinates:
    folium.CircleMarker(
        location=[passage["latitude"], passage["longitude"]],
        radius=5,
        color="red",
        fill=True,
        fill_color="red",
        popup=f"Altitude: {passage['altitude']}m\n{passage['timestamp']}"
    ).add_to(m)
folium.PolyLine(passage_5A_latlngs, color="red", weight=2, tooltip="Parcours 5A: B → A").add_to(m)

# Ajouter le trajet vert (A → B)
passage_5B_latlngs = [[p["latitude"], p["longitude"]] for p in passage_5B_coordinates]
for passage in passage_5B_coordinates:
    folium.CircleMarker(
        location=[passage["latitude"], passage["longitude"]],
        radius=5,
        color="green",
        fill=True,
        fill_color="green",
        popup=f"Altitude: {passage['altitude']}m\n{passage['timestamp']}"
    ).add_to(m)
folium.PolyLine(passage_5B_latlngs, color="green", weight=2, tooltip="Parcours 5B: A → B").add_to(m)

# Ajouter une légende
legend_html = """
<div style="
    position: fixed; 
    bottom: 20px; left: 20px; width: 270px; height: 140px; 
    background-color: white; z-index: 9999; font-size:14px;
    padding: 10px; border-radius: 5px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
">
<b>Légende :</b><br>
<span style="color: red;">■</span> Parcours 5A (B → A)<br>
<span style="color: green;">■</span> Parcours 5B (A → B)<br>

</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Ajouter des outils de dessin
draw = Draw(
    draw_options={ 
        'polyline': True,
        'rectangle': True,
        'polygon': True,
        'circle': True,
        'marker': True,
        'circlemarker': True
    },
    edit_options={'edit': True}
)
m.add_child(draw)

# Sauvegarder la carte HTML en mémoire
data = io.BytesIO()
m.save(data, close_file=False)
html_content = data.getvalue().decode()

# Trouver la variable de la carte Folium
match = re.search(r"var\s+(map_\w+)\s*=\s*L\.map", html_content)
if match:
    map_variable = match.group(1)
else:
    print("Erreur : Impossible de trouver la variable de la carte Folium.")
    sys.exit(1)

# JavaScript pour capturer les événements de dessin
injected_script = f"""
<script>
document.addEventListener("DOMContentLoaded", function() {{
    {map_variable}.on("draw:created", function(event) {{
        var layer = event.layer;
        var geoJson = layer.toGeoJSON();

        if (geoJson.geometry.type === "Point") {{
            var newCoords = geoJson.geometry.coordinates;
            var latlng = [newCoords[1], newCoords[0]];

            var name = prompt("Nom du point d'accès :", "AP_Nouveau");
            if (name) {{
                var dataToSend = JSON.stringify({{
                    "name": name,
                    "latitude": latlng[0],
                    "longitude": latlng[1]
                }}); 
                console.log(dataToSend);
            }}
        }} else {{
            console.error("Le type de géométrie n'est pas un point :", geoJson.geometry.type);
        }}
    }}); 
}});
</script>
"""

# Insérer le script dans l'HTML de la carte
modified_html = html_content.replace("</body>", injected_script + "</body>")

# Classe pour intercepter les messages de la console JavaScript
class WebEnginePage(QtWebEngineWidgets.QWebEnginePage):
    def javaScriptConsoleMessage(self, level, msg, line, sourceID):
        try:
            new_point = json.loads(msg)
            print("Nouveau point ajouté :", new_point)
            data = load_json(json_file_path)
            data.append(new_point)
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            print("Erreur JSON.")

# Création de l'interface Qt
view = QtWebEngineWidgets.QWebEngineView()
view.setPage(WebEnginePage(view))
view.setHtml(modified_html)
view.show()

sys.exit(app.exec_())
