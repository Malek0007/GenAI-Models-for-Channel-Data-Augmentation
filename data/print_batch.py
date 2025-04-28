import numpy as np

# Load the batch file
batch = np.load('data/batchs/batch_000.npy')  # Update path if needed

# Print basic info
print("Batch shape:", batch.shape)
print("Data type:", batch.dtype)

# Print first sample (index 0)
print("\nFirst sample:")
print(batch[0])  # shape: (128, 52, 8, 4)

# Optional: print a smaller slice
print("\nSample[0][0][0]:", batch[0, 0, 0, :, :])  # AP 0–7, user 0–3
