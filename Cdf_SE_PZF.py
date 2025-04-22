import json
import numpy as np
import matplotlib.pyplot as plt

# Étape 1 : Charger les données JSON depuis un fichier
with open("simulation_data.json", "r") as file:
    data = json.load(file)

# Étape 2 : Initialiser les matrices 8x10
SE_simule_matrix = np.zeros((8, 10))
SE_theorique_matrix = np.zeros((8, 10))

# Étape 3 : Remplir les matrices
counter = [0]*8  # Pour suivre combien de valeurs ont été remplies pour chaque UOIindex

for entry in data:
    uoi = entry["UOIindex"]
    if counter[uoi] < 10:  # max 10 valeurs par UOIindex
        SE_simule_matrix[uoi][counter[uoi]] = entry["SE_simule_PZF"]
        SE_theorique_matrix[uoi][counter[uoi]] = entry["SE_theorique_PZF"]
        counter[uoi] += 1

# Étape 4 : Aplatir les matrices pour la CDF
SE_simule_flat = SE_simule_matrix.flatten()
SE_theorique_flat = SE_theorique_matrix.flatten()

# Étape 5 : Trier pour la CDF
simule_sorted = np.sort(SE_simule_flat)
theorique_sorted = np.sort(SE_theorique_flat)

# Étape 6 : Générer les valeurs de CDF
cdf = np.linspace(0, 1, len(simule_sorted))

# Étape 7 : Tracer les CDF
plt.figure(figsize=(10, 6))
plt.plot(simule_sorted, cdf, label='SE Simulé PZF', linewidth=2)
plt.plot(theorique_sorted, cdf, label='SE Théorique PZF', linewidth=2, linestyle='--')

# Annotation pour indiquer le type de précodage
plt.text(0.1, 0.9, 'Précodage: PZF', transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'))

plt.title("CDF de SE (Spectral Efficiency) pour PZF")
plt.xlabel("Spectral Efficiency (SE)")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cdf_SE_pzf.png")
plt.show()

# Étape 8 : Sauvegarder les matrices dans un fichier JSON
se_matrices = {
    "SE_simule_PZF": SE_simule_matrix.tolist(),
    "SE_theorique_PZF": SE_theorique_matrix.tolist()
}

with open("se_matrices_pzf.json", "w") as outfile:
    json.dump(se_matrices, outfile, indent=4)
