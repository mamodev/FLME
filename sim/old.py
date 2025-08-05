
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

from model import SimpleCNN, GPULoader

def make_slots(device, NetFact, n_slots):
    slots = []
    for s in range(n_slots):
        stream = torch.cuda.Stream(device)
        net = NetFact() 
        with torch.cuda.stream(stream):
            net = NetFact().to(device, non_blocking=True)
        slots.append((stream, net))
    return slots

nworkers = 2
device = torch.device("cuda:0")
slots = make_slots(device, SimpleCNN, nworkers)

executor = ThreadPoolExecutor(nworkers)

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root=".data", transform=transform, train=True, download=True)


slots = [(stream, net, GPULoader(train_dataset, 512, False, device, stream)) for stream, net in slots]


def train_on_slot(slot):
    stream=slot[0]
    net=slot[1]
    train_loader = slot[2]
    with torch.cuda.stream(stream):
        net.train()
        
        lr = 0.01
        mo = 0.001
        wd = 0.01

        optimizer = torch.optim.SGD(net.parameters(),
                                        lr=lr, momentum=mo, weight_decay=wd)


        # accumulate GPU‐side loss
        total_loss = torch.zeros((), device=device)
        print("start training")
        for X, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out   = net(X)
            loss = F.cross_entropy(out, y, reduction="mean")
            loss.backward()
            optimizer.step()
            total_loss += loss

        # average loss tensor
        avg_loss = total_loss / len(train_loader)

        print(f"avg loss: {avg_loss}")

        done_evt = torch.cuda.Event()
        done_evt.record(stream)
        
    done_evt.synchronize()


while True:
    futures = []
    for slot in slots:
       f = executor.submit(train_on_slot, slot)
       futures.append(f)

    for fut in futures:
        ign = fut.result()
        
