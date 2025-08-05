# GPU Hang Recovery

If your PyTorch script hangs indefinitely on even a tiny `tensor.to("cuda:0")` call, your GPU driver (often the NVIDIA MPS server) may be wedged. Add this note to your project README or troubleshooting guide.

## Symptoms

- `x.to("cuda:0")` blocks forever
- No obvious Python traceback
- You may have seen errors mentioning “MCP server” or `nvidia-cuda-mps-server`

## Quick Recovery Steps

1. Kill any stale MPS server  
   ```bash
   pkill -9 nvidia-cuda-mps-server
   ```
2. Remove the old MPS pipe/log directory  
   ```bash
   rm -rf /tmp/nvidia-mps
   ```
3. Verify no GPU contexts remain  
   ```bash
   nvidia-smi
   ```
   You should see no Python or other processes using the GPU.
4. Test a minimal transfer to confirm recovery:  
   ```python
   import torch

   print("CUDA available:", torch.cuda.is_available())
   x = torch.tensor([1])
   print("Attempting transfer…")
   y = x.to("cuda:0")
   print("Succeeded:", y)
   ```
5. If it still hangs, you can either:
   - Unload/reload the NVIDIA kernel modules (requires root, no other GPU users):
     ```bash
     sudo systemctl stop nvidia-persistenced
     sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia
     sudo modprobe nvidia nvidia_uvm nvidia_modeset nvidia_drm
     sudo systemctl start nvidia-persistenced
     ```
   - Or reboot the machine.

## (Re)Starting MPS

Only if you really need MPS:
```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
nvidia-cuda-mps-control -d
```

## Best Practices to Avoid Future Hangs

- In your training loop, move **batches**, not the entire dataset:
  ```python
  loader = DataLoader(
      dataset, batch_size=64,
      shuffle=True, num_workers=4,
      pin_memory=True
  )
  for xb, yb in loader:
      xb = xb.to(device, non_blocking=True)
      yb = yb.to(device, non_blocking=True)
      …
  ```
- Use a reasonable `num_workers` (≤ number of physical CPU cores, typically 2–8).
- At program entry (under `if __name__ == "__main__":`), set the multiprocessing start method to `spawn`:
  ```python
  import torch.multiprocessing as mp
  mp.set_start_method('spawn', force=True)
  ```