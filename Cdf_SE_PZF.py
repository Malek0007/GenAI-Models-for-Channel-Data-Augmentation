import json
import numpy as np
import matplotlib.pyplot as plt

def process_file(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    SE_simule_matrix = np.zeros((8, 10))
    SE_theorique_matrix = np.zeros((8, 10))
    counter = [0]*8

    for entry in data:
        uoi = entry["UOIindex"]
        if counter[uoi] < 10:
            SE_simule_matrix[uoi][counter[uoi]] = entry["SE_simule_PZF"]
            SE_theorique_matrix[uoi][counter[uoi]] = entry["SE_theorique_PZF"]
            counter[uoi] += 1

    return SE_simule_matrix, SE_theorique_matrix

# 🔄 Traitement des fichiers
SE_simule_matrix_PZF, SE_theorique_matrix_PZF = process_file("simulation_data_PZF.json")
SE_simule_matrix_MRT, SE_theorique_matrix_MRT = process_file("simulation_data_MRT.json")

# 🔀 Conversion en tableaux plats et tri
def sorted_cdf(matrix):
    sorted_vals = np.sort(matrix.flatten())
    cdf_vals = np.linspace(0, 1, len(sorted_vals))
    return sorted_vals, cdf_vals

simule_sorted_PZF, cdf_PZF = sorted_cdf(SE_simule_matrix_PZF)
theorique_sorted_PZF, _ = sorted_cdf(SE_theorique_matrix_PZF)
simule_sorted_MRT, cdf_MRT = sorted_cdf(SE_simule_matrix_MRT)
theorique_sorted_MRT, _ = sorted_cdf(SE_theorique_matrix_MRT)

# 📊 Tracer toutes les courbes sur une seule figure
plt.figure(figsize=(12, 7))

plt.plot(simule_sorted_PZF, cdf_PZF, label='SE Simulé PZF', linewidth=2)
plt.plot(theorique_sorted_PZF, cdf_PZF, label='SE Théorique PZF', linewidth=2, linestyle='--')
plt.plot(simule_sorted_MRT, cdf_MRT, label='SE Simulé MRT', linewidth=2)
plt.plot(theorique_sorted_MRT, cdf_MRT, label='SE Théorique MRT', linewidth=2, linestyle='--')

plt.text(3, 0.6, "Precodage MRT", fontsize=12, fontweight='bold', color='blue')
plt.text(5.5, 0.6, "Precodage PZF", fontsize=12, fontweight='bold', color='darkgreen')

plt.title("CDF de SE (Spectral Efficiency) - PZF vs MRT")
plt.xlabel("Spectral Efficiency (SE)")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cdf_SE_comparaison_PZF_MRT.png")
plt.show()

# 💾 Sauvegarde des matrices
matrices_dict = {
    "PZF": {
        "SE_simule_matrix": SE_simule_matrix_PZF.tolist(),
        "SE_theorique_matrix": SE_theorique_matrix_PZF.tolist()
    },
    "MRT": {
        "SE_simule_matrix": SE_simule_matrix_MRT.tolist(),
        "SE_theorique_matrix": SE_theorique_matrix_MRT.tolist()
    }
}

with open("se_matrices_combined.json", "w") as f:
    json.dump(matrices_dict, f, indent=4)

print("✅ Matrices PZF et MRT sauvegardées dans 'se_matrices_combined.json'")
