import json
from xml.etree import ElementTree as ET

def extract_coordinates(kml_file):
    # Charger le fichier KML
    tree = ET.parse(kml_file)
    root = tree.getroot()

    # Espace de noms KML
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    coordinates_list = []
    
    # Parcourir les balises Placemark pour extraire les coordonnées
    for placemark in root.findall(".//kml:Placemark", ns):
        name = placemark.find("kml:name", ns)
        coords = placemark.find(".//kml:coordinates", ns)
        
        if coords is not None:
            lon, lat, _ = coords.text.strip().split(",")  # Séparer longitude, latitude, altitude
            coordinates_list.append({
                "name": name.text if name is not None else "Unknown",
                "latitude": float(lat),
                "longitude": float(lon)
            })
    
    return coordinates_list

def save_to_json(data, json_file):
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Données enregistrées dans {json_file}")

# Utilisation
kml_file_path = "geodata/2022-10-06_AP_Position.kml"
json_file_path = "AP_coordinates.json"

coordinates = extract_coordinates(kml_file_path)
save_to_json(coordinates, json_file_path)
