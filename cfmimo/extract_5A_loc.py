import xml.etree.ElementTree as ET
import json

# Charger le fichier KML
def extract_coordinates_from_kml(kml_file, json_file):
    tree = ET.parse(kml_file)
    root = tree.getroot()
    
    # Définition de l'espace de noms
    ns = {'kml': 'http://www.opengis.net/kml/2.2', 'gx': 'http://www.google.com/kml/ext/2.2'}
    
    # Recherche des coordonnées
    coordinates = []
    for track in root.findall(".//gx:Track", ns):
        for coord, timestamp in zip(track.findall("gx:coord", ns), track.findall("kml:when", ns)):
            lon, lat, alt = map(float, coord.text.split())
            coordinates.append({
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "timestamp": timestamp.text
            })
    
    # Sauvegarde au format JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(coordinates, f, indent=4)

# Exemple d'utilisation
extract_coordinates_from_kml("geodata/Passage_5A.kml", "Passage_5A.json")