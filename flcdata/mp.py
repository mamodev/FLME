import time
def dumb_worker(in_ch):
    tensor = in_ch.get()

    # try to touch data keeping it in device memory
    tensor.sum().item()

    try:
        time.sleep(100000)
    except:
        pass
    
import torch
import torch.multiprocessing as mp




if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    n_devices = torch.cuda.device_count()
    worker_per_device = 8
    devices = [torch.device(f"cuda:{i}") for i in range(n_devices)]

    tensor_size = 1024 * 1024 * 1024 // 4  # 1 GB tensor
    # tensor_size = 1  # 4 bytes tensor for testing
    
    
    device_data = []
    for i in range(n_devices):
        fake_tensor = torch.zeros((1, tensor_size), device=devices[i])
        device_data.append(fake_tensor)
        
        
    q = mp.Queue()
    for d in device_data:
        for _ in range(worker_per_device):
            q.put(d) #pythotch shold handle this IPC
    
    del device_data  # Free the list to avoid holding references to tensors
    
    procs = []
    for device in devices:
        for _ in range(worker_per_device):
            p = mp.Process(target=dumb_worker, args=(q,))
            p.start()
            procs.append(p)
            
    for p in procs:
        p.join()

