import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid")  


# Charger les fichiers
with open("simulation_data_MRT_8.json") as f:
    mrt_data = json.load(f)

with open("simulation_data_PZF_8.json") as f:
    pzf_data = json.load(f)

with open("simulation_data_FZF_8.json") as f:
    fzf_data = json.load(f)

# Extraire les SE simulés et théoriques
SE_mrt_sim = [d["SE_simule_PZF"] for d in mrt_data]
SE_mrt_theo = [d["SE_theorique_PZF"] for d in mrt_data]

SE_pzf_sim = [d["SE_simule_PZF"] for d in pzf_data]
SE_pzf_theo = [d["SE_theorique_PZF"] for d in pzf_data]

SE_fzf_sim = [d["SE_simule_PZF"] for d in fzf_data]
SE_fzf_theo = [d["SE_theorique_PZF"] for d in fzf_data]

se_data = {
    "MRT": {
        "SE_simule_PZF": SE_mrt_sim,
        "SE_theorique_PZF": SE_mrt_theo
    },
    "PZF": {
        "SE_simule_PZF": SE_pzf_sim,
        "SE_theorique_PZF": SE_pzf_theo
    },
    "FZF": {
        "SE_simule_PZF": SE_fzf_sim,
        "SE_theorique_PZF": SE_fzf_theo
    }
}

# Sauvegarder dans un fichier JSON
with open("se_matrices.json", "w") as json_file:
    json.dump(se_data, json_file, indent=4)
    
# Fonction pour calculer la CDF
def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

# Calculer les CDFs
x_mrt_sim, y_mrt_sim = compute_cdf(SE_mrt_sim)
x_mrt_theo, y_mrt_theo = compute_cdf(SE_mrt_theo)

x_pzf_sim, y_pzf_sim = compute_cdf(SE_pzf_sim)
x_pzf_theo, y_pzf_theo = compute_cdf(SE_pzf_theo)

x_fzf_sim, y_fzf_sim = compute_cdf(SE_fzf_sim)
x_fzf_theo, y_fzf_theo = compute_cdf(SE_fzf_theo)

# Tracer la CDF
plt.figure(figsize=(14, 8))

# MRT (Bleu)
plt.plot(x_mrt_sim, y_mrt_sim, label="SE simulé MRT", color="royalblue", linestyle='-', linewidth=2.5)
plt.plot(x_mrt_theo, y_mrt_theo, label="SE théorique MRT", color="royalblue", linestyle='--', linewidth=2)
plt.text(np.median(x_mrt_sim), 0.6, 'Précodage : MRT', fontsize=12, color='royalblue',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='royalblue', boxstyle='round'))

# PZF (Vert)
plt.plot(x_pzf_sim, y_pzf_sim, label="SE simulé PZF", color="forestgreen", linestyle='-', linewidth=2.5)
plt.plot(x_pzf_theo, y_pzf_theo, label="SE théorique PZF", color="forestgreen", linestyle='--', linewidth=2)
plt.text(np.median(x_pzf_sim), 0.5, 'Précodage : PZF', fontsize=12, color='forestgreen',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='forestgreen', boxstyle='round'))

# FZF (Violet)
plt.plot(x_fzf_sim, y_fzf_sim, label="SE simulé FZF", color="mediumorchid", linestyle='-', linewidth=2.5)
plt.plot(x_fzf_theo, y_fzf_theo, label="SE théorique FZF", color="mediumorchid", linestyle='--', linewidth=2)
plt.text(np.median(x_fzf_sim), 0.4, 'Précodage : FZF', fontsize=12, color='mediumorchid',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='mediumorchid', boxstyle='round'))

# Personnalisation
plt.xlabel("Spectral Efficiency (SE)", fontsize=14, fontweight='bold')
plt.ylabel("Fonction de Répartition Cumulative (CDF)", fontsize=14, fontweight='bold')
plt.title("CDF du SE (8 antennes) - Simulé vs Théorique", fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Légende stylée
plt.legend(fontsize=12, frameon=True, loc="lower right", fancybox=True, framealpha=0.9, shadow=True, borderpad=1)

# Marge et sauvegarde
plt.tight_layout()
plt.savefig("cdf_precodage_M8.png", dpi=300)
plt.show()
