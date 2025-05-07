import json
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# Chemin vers le fichier JSON
file_path = "matrix_4_8_52_818/performance_4_8_52_818.json"
output_image = "matrix_4_8_52_818/cdf_SE_4_8_52_818.png"  # Nom de l’image à sauvegarder

def load_SE_data(file_path):
    """Charge les SE depuis le fichier JSON"""
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            return [entry["SE_simule_PZF"] for entry in data if "SE_simule_PZF" in entry]
        except json.JSONDecodeError:
            return []

def plot_and_save_cdf(se_list, output_path):
    """Trace, enregistre et affiche la CDF"""
    if len(se_list) == 0:
        return

    se_sorted = np.sort(se_list)
    cdf = np.arange(1, len(se_sorted) + 1) / len(se_sorted)

    fig = plt.figure()
    plt.plot(se_sorted, cdf, label="SE simulé")
    plt.xlabel("Spectral Efficiency (bit/s/Hz)")
    plt.ylabel("CDF")
    plt.title("CDF de l'efficacité spectrale en temps réel (M=4, L=52)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    

    # ✅ Enregistre d'abord l'image
    fig.savefig(output_path)

    # ✅ Affiche ensuite la figure
    plt.show(block=False)
    plt.pause(100)  # Affiche pendant 1 seconde
    plt.close(fig)



# --- Boucle de mise à jour automatique ---
print("Lancement du suivi temps réel + sauvegarde des CDF... (Ctrl+C pour arrêter)")
try:
    previous_len = 0
    while True:
        se_data = load_SE_data(file_path)
        if len(se_data) != previous_len:
            previous_len = len(se_data)
            plot_and_save_cdf(se_data, output_image)
            print(f"Nouvelle CDF sauvegardée ({len(se_data)} points)")
        time.sleep(2)
except KeyboardInterrupt:
    print("Arrêt manuel du suivi.")
