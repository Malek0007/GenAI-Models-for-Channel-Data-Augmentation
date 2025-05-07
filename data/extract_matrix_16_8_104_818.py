import numpy as np
import h5py
import os

# Chemin vers le fichier .mat
mat_path = r'data\HMIMO_6GHz_Cellfree_Pass5A_8AP1.mat'

# Ouverture du fichier et lecture de 'HMIMO'
with h5py.File(mat_path, 'r') as f:
    hmimo = f['HMIMO']
    print("Dtype de 'HMIMO' :", hmimo.dtype)

    if hmimo.dtype.names is not None:
        print("Champs dans 'HMIMO' :", hmimo.dtype.names)
        real_data = np.array(hmimo['real'])
        imag_data = np.array(hmimo['imag'])
        data = real_data + 1j * imag_data
    else:
        data = np.array(hmimo)

print("Shape de data :", data.shape)
assert np.iscomplexobj(data), "La matrice n'est pas complexe !"

#########################################################################
# Fonction pour générer les indices
def generate_indices(start_index=0, n=16):
    return [start_index + 4 * i for i in range(n)]

# Extraction 1er AP : symboles à partir de l'indice 0
indices_1 = generate_indices(start_index=0)
raw_selected_1 = data[:, indices_1, :, :, 0]  # (818, 8, 52, 16)
print("Shape finale de la matrice extraite1 :", raw_selected_1.shape)
print(raw_selected_1[0, 0, :, 0])

# Transposition vers (16, 8, 52, 818)

selected_1 = raw_selected_1.transpose(0, 3, 2, 1) 

# Extraction 2ème AP : symboles à partir de l’indice 64
indices_2 = generate_indices(start_index=64)
raw_selected_2 = data[:, indices_2, :, :, 0]  
print("Shape finale de la matrice extraite2 :", raw_selected_2.shape)
print(raw_selected_2[0, 0, :, 0])
# Transposition
selected_2 = raw_selected_2.transpose(0, 3, 2, 1)  

#########################################################################

# Concaténation sur l’axe des sous-porteuses (axe=2)
combined_matrix = np.concatenate([selected_1, selected_2], axis=2)  # (16, 8, 104, 818)

#########################################################################
# Sauvegarde
output_dir = 'matrix2_16_8_104_818'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'matrix_16_8_104_818.npy')
np.save(output_path, combined_matrix)
print(f"Matrice combinée sauvegardée à : {output_path}")

#########################################################################
# Vérification
print("Shape finale de la matrice combinée :", combined_matrix.shape)

# Affichage de quelques valeurs
print("\nValeurs [symbol=0, user=0, :, subcarrier=0] (soit 104 trames concaténées) :")
print(combined_matrix[0, 0, :, 0])
