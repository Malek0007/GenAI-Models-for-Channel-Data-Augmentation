import h5py
import numpy as np
import pandas as pd

# Load the .mat file
mat_file = r'd:\Users\mabida\Downloads\GenAI-Models-for-Channel-Data-Augmentation-main\data\HMIMO_6GHz_Cellfree_Pass5A_8AP1.mat'

with h5py.File(mat_file, 'r') as f:
    # Extract the real and imaginary parts of the complex dataset
    data = f['HMIMO']
    real = data['real']
    imag = data['imag']
    complex_data = real[...] + 1j * imag[...]

    # Flatten each sample into 1D and combine into a 2D array
    num_samples = complex_data.shape[0]
    flat_data = complex_data.reshape(num_samples, -1)  # shape: (818, N)

    # Convert complex numbers to strings like "a+bj"
    flat_data_str = flat_data.astype(str)

    # Save to CSV
    df = pd.DataFrame(flat_data_str)
    df.to_csv('data\hmimo_Passage5A_complex.csv', index=False)
