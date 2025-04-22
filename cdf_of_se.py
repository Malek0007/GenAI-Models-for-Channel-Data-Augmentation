import json
import numpy as np
import matplotlib.pyplot as plt

# Charger les données SEksim0 (simulées)
with open("Performance_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

num_users = 8
num_snapshots = len(data) // num_users
SEksim0_matrix = np.zeros((num_users, num_snapshots))

for i, entry in enumerate(data):
    user_index = entry["user"] - 1
    snapshot_index = i // num_users
    SEksim0_matrix[user_index, snapshot_index] = entry["SEksim0"]

SEksim0_flat = SEksim0_matrix.flatten()

# Charger les données SE_theo (théoriques)
with open("data_performance1.json", "r", encoding="utf-8") as f:
    data_theo = json.load(f)

SE_theo_matrix = np.zeros((num_users, num_snapshots))

for i, entry in enumerate(data_theo):
    user_index = entry["Utilisateur"]
    snapshot_index = i // num_users
    SE_theo_matrix[user_index, snapshot_index] = entry["SE_theo"]

SE_theo_flat = SE_theo_matrix.flatten()

# Fonction CDF
def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

x_sim, cdf_sim = compute_cdf(SEksim0_flat)
x_theo, cdf_theo = compute_cdf(SE_theo_flat)

# Tracer
plt.figure(figsize=(8, 5))
plt.plot(x_sim, cdf_sim, label="SE simulé (PZF)", color="red", linewidth=2)
plt.plot(x_theo, cdf_theo, label="SE théorique (PZF)", color="green", linewidth=2, linestyle="--")

plt.xlabel("Score d'Erreur Spectral (SE)")
plt.ylabel("CDF")
plt.title("CDF de SE simulé et SE théorique - Precoding PZF")
plt.legend()
plt.grid(True)

# Ajouter annotation de la méthode
plt.text(0.5, 0.1, "Méthode : Precoding PZF", transform=plt.gca().transAxes,
         fontsize=10, color="black", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()
