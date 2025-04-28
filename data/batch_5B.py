import h5py
import numpy as np
import os

# Paths
mat_path = r'd:\Users\mabida\Downloads\GenAI-Models-for-Channel-Data-Augmentation-main\data\HMIMO_6GHz_Cellfree_Pass5A_8AP1.mat'
batch_dir = r'd:\Users\mabida\Downloads\GenAI-Models-for-Channel-Data-Augmentation-main\data\batchs'

# Create the batch directory if it doesn't exist
os.makedirs(batch_dir, exist_ok=True)

# Open the .mat file
with h5py.File(mat_path, 'r') as f:
    data = f['HMIMO']
    batch_size = 10
    num_samples = data.shape[0]

    for i in range(0, num_samples, batch_size):
        # Slice 'real' and 'imag'
        real_batch = data['real'][i:i+batch_size]
        imag_batch = data['imag'][i:i+batch_size]

        # Combine to complex
        complex_batch = real_batch + 1j * imag_batch

        # Save batch to .npy in the 'batchs' directory
        filename = os.path.join(batch_dir, f'batch_{i:03}.npy')
        np.save(filename, complex_batch)

        print(f"Saved {filename}")
