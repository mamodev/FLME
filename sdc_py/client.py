import argparse
import socket
import struct
import jwt
import protocol
import datetime

import time

from typing import Tuple

import torch
from torch.utils.data import DataLoader

SECRET_KEY = "some_secret_key"


def deepCloneToCpu(model):
    return {k: v.detach().clone().cpu() for k, v in model.items()}

def dataset_to_device(dataset, device, batch_size=1024, shuffle=False):
    assert dataset.data.device != device, f"Dataset already on device: {dataset.data.device}"
    assert dataset.targets.device != device, f"Targets already on device: {dataset.targets.device}"

    dataset.data = dataset.data.to(device)
    dataset.targets = dataset.targets.to(device)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# example of cli: python client.py PORT
parser = argparse.ArgumentParser(description='Client for SDC')
parser.add_argument('port', type=int, help='Port to connect to')
args = parser.parse_args()

# create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', args.port))

utc_now = datetime.datetime.now(datetime.timezone.utc)

jwt_token = jwt.encode({
    'key': 'some_uuid',
    'exp': utc_now + datetime.timedelta(hours=1),
}, SECRET_KEY, algorithm='HS256')

jwt_token = jwt_token.encode('utf-8')


def recvAll(s: socket.socket, n: int) -> bytes:
    data = b''
    while len(data) < n:
        packet = s.recv(n - len(data))
        if not packet:
            raise Exception("Connection closed")
        data += packet
    return data

def recv_packet(s: socket.socket) -> Tuple[int, bytes]:
    packet_id = recvAll(s, 2)
    packet_len = recvAll(s, 4)
    packet_len = struct.unpack('!I', packet_len)[0]
    payload = recvAll(s, packet_len)

    return int.from_bytes(packet_id, 'big'), payload

# send auth packet
s.send(protocol.create_packet(protocol.AuthPacketID, jwt_token))
packet_id, payload = recv_packet(s)
if packet_id != 0:
    print(f"Failed to authenticate: {payload}")
    s.close()
    exit(1)

from model import Model
from cubeds import CuboidDataset
path = "./data/cuboids/cuboids"
train_dss = CuboidDataset.LoadGroups(ds_folder=path, train=True)
test_ds = CuboidDataset.LoadMerged(ds_folder=path, train=False)

train_loaders = [dataset_to_device(ds, 'cuda') for ds in train_dss]
test_loader = dataset_to_device(test_ds, 'cuda')

lr = 0.001
momentum = 0 
ephocs = 3

idx = 0

while True:
    idx = (idx + 1) % len(train_loaders)
    train_loader = train_loaders[idx]

    net = Model()
    net.to('cuda')
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    model_state = net.state_dict()
    model_state = {k: v.detach().clone() for k, v in model_state.items()}

    # send get model packet
    s.send(protocol.create_packet(protocol.GetModelPacketID, protocol.GetModelPacket(0).to_buffer()))
    packet_id, payload = recv_packet(s)
    if packet_id != 0:
        print(f"Failed to get model: {payload}")
        s.close()
        exit(1)

    MODEL_VERSION = struct.unpack('!I', payload[:4])[0]
    model_data, _ = protocol.ModelData.from_buffer(payload[4:])
    print(f"Model version: {MODEL_VERSION}, bytes: {len(payload)}")

    net.load_state_dict(model_data.model_state)
    for e in range(ephocs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()


    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f"Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)")


    print("Send updated model")
    model_state = deepCloneToCpu(net.state_dict())

    model_meta = protocol.ModelMeta(
        momentum=momentum,
        learning_rate=lr,
        train_loss=0.1,
        test_loss=0.2,
        local_epoch=ephocs,
        train_samples=len(train_dss[idx]),
        derived_from=MODEL_VERSION
    )

    model_data = protocol.ModelData(model_state)
    putModelPacket = protocol.PutModelPacket(model_data, model_meta)
    putModelPacketBuffer = putModelPacket.to_buffer()

    s.send(protocol.create_packet(protocol.PutModelPacketID, putModelPacketBuffer))

    print("waiting for response")
    packet_id, payload = recv_packet(s)
    if packet_id != 0:
        print(f"Failed to put model: {payload}")
        s.close()
        exit(1)

    print("Model sent successfully")

    time.sleep(0.4)