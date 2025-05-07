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
# Indices de symboles
def generate_indices(start_index=0, n=16):
    return [start_index + 4 * i for i in range(n)]

indices = generate_indices(start_index=0)

# Extraction : shape (818, 16, 52, 8)
raw_selected = data[:, indices, :, :, 0]
print("Shape finale de la matrice extraite :", raw_selected.shape)
print(raw_selected[0, 0, :, 0])

# Transposition vers (16, 8, 52, 818)
selected_matrix = raw_selected.transpose(0, 3, 2, 1)

#########################################################################
# Sauvegarde
output_dir = 'matrix_16_8_52_818'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'matrix_16_8_52_818.npy')
np.save(output_path, selected_matrix)
print(f"Matrice transposée sauvegardée à : {output_path}")

#########################################################################
# Vérification
print("Shape finale de la matrice transposée :", selected_matrix.shape)

# Affichage de quelques valeurs
print("\nValeurs [symbol=0, user=0, :, subcarrier=0] (soit 52 trames) :")
print(selected_matrix[0, 0, :, 0])
