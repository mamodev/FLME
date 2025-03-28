import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Matrix size
n = 8192

# Create random matrices on the GPU
a = torch.randn(n, n, device=device)
b = torch.randn(n, n, device=device)

# Warm-up (run a few times to make sure everything is loaded)
for _ in range(5):
    c = torch.matmul(a, b)

# Time the matrix multiplication
start_time = time.time()
for _ in range(2000):
    c = torch.matmul(a, b)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
print(f"Operations per second: {10 / elapsed_time:.2f}")