import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#!/usr/bin/env python3
import signal
import mmap
import posix_ipc
import numpy as np
import gc
import struct
import torch

import sys
import os
import argparse
from lib.flcdata import Model
import traceback

def signal_handler(signum, frame):
    # print(f"Python: signal {signum} received, exiting")
    pass

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGQUIT, signal_handler)

__CLEANUP_FUNCS__ = []
def cleanup():
    # print("Python: cleaning up")
    gc.collect()
    __CLEANUP_FUNCS__.reverse()
    for fn, dex in __CLEANUP_FUNCS__:
        try:
            fn()
        except Exception as e:
            print(f"Python: error cleaning up {dex} : {e}")

def safe_shared_memory(name, size):
    shm = posix_ipc.SharedMemory(name)
    __CLEANUP_FUNCS__.append((lambda: shm.close_fd(), f"shared_memory::close_fd name={name}"))
    return shm

def safe_mmap(fd, size, prot):
    mm = mmap.mmap(fd, size, prot=prot)
    __CLEANUP_FUNCS__.append((lambda: mm.close(), f"mmap::close fd={fd}"))
    return mm

def safe_mqueue(name):
    mq = posix_ipc.MessageQueue(name)
    __CLEANUP_FUNCS__.append((lambda: mq.close(), f"mqueue::close name={name}"))
    return mq

# struct ptc_msg {
#     uint32_t id;
#     int32_t err_code;
# };

# struct ctp_msg {
#     uint32_t id;

#     uint64_t model_offset;
#     uint64_t dataset_offset;
#     uint64_t results_offset;

#     uint32_t model_size;
#     uint32_t partition;
#     uint32_t partition_offset;
#     uint32_t partition_size;
# };


# def CustomDataLoader(data, targets, batch_size, shuffle):
class CustomDataLoader:
    def __init__(self, data, targets, batch_size, shuffle):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        if shuffle:
            self.indices = np.arange(len(data))
            np.random.shuffle(self.indices)

    def __iter__(self):
        if self.shuffle:
            for i in range(0, len(self.data), self.batch_size):
                indices = self.indices[i:i+self.batch_size]
                batch_data = self.data[indices]
                batch_targets = self.targets[indices]
                yield batch_data, batch_targets
        else:
            for i in range(0, len(self.data), self.batch_size):
                batch_data = self.data[i:i+self.batch_size]
                batch_targets = self.targets[i:i+self.batch_size]
                yield batch_data, batch_targets


def compute(mm_models, mm_dataset, mm_results,
                model_offset,
                result_offset,
                model_size,
                data_offset,
                data_shape,
                targets_offset,
                targets_shape,
                hyperparameters):

    mm_results[result_offset:result_offset+model_size] = mm_models[model_offset:model_offset+model_size]

    mm_results.seek(result_offset)

    model_size = struct.unpack("i", mm_results.read(4))[0]
    # print(f"Python: model size = {model_size}")

    num_layers = struct.unpack("i", mm_results.read(4))[0]
    # print(f"Python: num layers = {num_layers}")

    model = {}
    layers = []

    for i in range(num_layers):
        key_len = struct.unpack("i", mm_results.read(4))[0]
        key = mm_results.read(key_len).decode('ascii')
        shape_len = struct.unpack("i", mm_results.read(4))[0]
        shape = []
        for j in range(shape_len):
            dim = struct.unpack("i", mm_results.read(4))[0]
            shape.append(dim)

        layers.append((key, shape))


    boffset = mm_results.tell()
    shm_layers_offset = boffset - result_offset
    for i in range(num_layers):
        data = np.frombuffer(
            mm_results,
            dtype=np.float32,
            count=np.prod(layers[i][1]),
            offset=boffset  
        )

        boffset += data.nbytes

        data = data.reshape(layers[i][1])
        model[layers[i][0]] = torch.from_numpy(data)
        del data

    mm_dataset.seek(0)
    dataset_size = struct.unpack("I", mm_dataset.read(4))[0]
    npartitions = struct.unpack("I", mm_dataset.read(4))[0]
    nfeatures = struct.unpack("I", mm_dataset.read(4))[0]
    nclasses = struct.unpack("I", mm_dataset.read(4))[0]

    data = np.frombuffer(
        mm_dataset,
        dtype=np.float32,
        count=np.prod(data_shape),
        offset=data_offset
    )


    targets = np.frombuffer(
        mm_dataset,
        dtype=np.int64,
        count=np.prod(targets_shape),
        offset=targets_offset
    )

    data = data.reshape(data_shape)
    targets = targets.reshape(targets_shape)

    tensor_data = None
    tensor_targets = None

    # print(f"Python: nfeatures = {nfeatures}, nclasses = {nclasses}")
    net = Model(insize=nfeatures, outsize=nclasses)

    for full_name, _ in list(net.named_parameters()):
        parent_name, param_name = full_name.rsplit('.', 1)
        parent = net
        for sub in parent_name.split('.'):
            parent = getattr(parent, sub)
        
        del parent._parameters[param_name]

    for k in model.keys():
        p = torch.nn.Parameter(model[k], requires_grad=True)
        parent_name, param_name = k.rsplit('.', 1)
        parent = net
        for sub in parent_name.split('.'):
            parent = getattr(parent, sub)
        
        parent.register_parameter(param_name, p)
  
    tensor_data = torch.from_numpy(data)
    tensor_targets = torch.from_numpy(targets)

    loader = CustomDataLoader(tensor_data, tensor_targets, hyperparameters["batch_size"], hyperparameters["shuffle"])
    
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), 
                    lr=hyperparameters["learning_rate"], 
                    momentum=hyperparameters["momentum"], 
                    weight_decay=hyperparameters["weight_decay"])
    
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(hyperparameters["ephocs"]):
        for batch_data, batch_targets in loader:
            optimizer.zero_grad()
            outputs = net(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

    # write back the model to the shared memory
    # print(f"Python: writing back model to shm_results at offset {result_offset} + {shm_layers_offset}, size {model_size}")
    # mm_results.seek(result_offset + shm_layers_offset)
    # mm_results.write(b'\x01' * (model_size - shm_layers_offset))

    # model = net.state_dict()
    # offs = result_offset + shm_layers_offset
    # for layer in layers:
    #     key, shape = layer
    #     data = model[key].detach().numpy()
    #     # mm_results[offs:offs+data.nbytes] = data.tobytes()
    #     mm_results.seek(offs)
    #     mm_results.write(data.tobytes())
    #     offs += data.nbytes

    # net.eval()
    # with torch.no_grad():
    #     outputs = net(tensor_data)
    #     _, predicted = torch.max(outputs.data, 1)
    #     correct = (predicted == tensor_targets).sum().item()
    #     total = tensor_targets.size(0)
    #     accuracy = 100 * correct / total
    #     print(f"Python: Accuracy of the model on the test data: {accuracy:.2f}%")

def main():
    p = argparse.ArgumentParser(description="POSIX MQ worker")
    p.add_argument("--in-queue",  required=True, help="name of input queue")
    p.add_argument("--out-queue", required=True, help="name of output queue")
    
    p.add_argument("--shm-models", required=True, help="name of model shm")
    p.add_argument("--shm-dataset", required=True, help="name of dataset shm")
    p.add_argument("--shm-results", required=True, help="name of results shm")

    p.add_argument("--shm-models-size", required=True, type=int, help="size of model shm")
    p.add_argument("--shm-dataset-size", required=True, type=int, help="size of dataset shm")
    p.add_argument("--shm-results-size", required=True, type=int, help="size of results shm")

    args = p.parse_args()

    IN_QUEUE   = args.in_queue
    OUT_QUEUE  = args.out_queue
    IN_FMT = "<QQQIQI16IQI16IIIfff?"
    IN_SIZE = struct.calcsize(IN_FMT)

    OUT_FMT = "<QiQQ"
    OUT_SIZE = struct.calcsize(OUT_FMT)

    try:
        mq_in  = safe_mqueue(IN_QUEUE)
        mq_out = safe_mqueue(OUT_QUEUE)
        shm_models = safe_shared_memory(args.shm_models, args.shm_models_size)
        shm_dataset = safe_shared_memory(args.shm_dataset, args.shm_dataset_size)
        shm_results = safe_shared_memory(args.shm_results, args.shm_results_size)

        mm_models = safe_mmap(shm_models.fd, args.shm_models_size, mmap.PROT_READ)
        mm_dataset = safe_mmap(shm_dataset.fd, args.shm_dataset_size, mmap.PROT_READ | mmap.PROT_WRITE)
        mm_results = safe_mmap(shm_results.fd, args.shm_results_size, mmap.PROT_WRITE | mmap.PROT_READ)
    except Exception as e:
        print(f"Python: error opening queues or shared memory: {e}")
        cleanup()
        sys.exit(1)                


    while True:
        # print(f"Python: waiting on {IN_QUEUE} (fmt={IN_FMT})")
        msg_bytes, _ = mq_in.receive(IN_SIZE)
        assert len(msg_bytes) == IN_SIZE, f"Python: invalid message size {len(msg_bytes)} != {IN_SIZE}"
        # print(f"Python: received {len(msg_bytes)} bytes")
        raw_in = struct.unpack(IN_FMT, msg_bytes)

        id = raw_in[0]
        model_offset = raw_in[1]
        result_offset = raw_in[2]
        model_size = raw_in[3]

        raw_in = raw_in[4:]


        data_offset = raw_in[0]
        raw_in = raw_in[1:]

        data_shape_len = raw_in[0]
        data_shape = []
        for j in range(data_shape_len):
            data_shape.append(raw_in[1+j])

        raw_in = raw_in[17:]
        
        targets_offset = raw_in[0]
        raw_in = raw_in[1:]

        targets_shape_len = raw_in[0]
        targets_shape = []
        for j in range(targets_shape_len):
            targets_shape.append(raw_in[1+j])

        raw_in = raw_in[17:]

        ephocs = raw_in[0]
        batch_size = raw_in[1]
        learning_rate = raw_in[2]
        momentum = raw_in[3]
        weight_decay = raw_in[4]
        shuffle = raw_in[5]

        hyperparameters = {
            "ephocs": ephocs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "shuffle": shuffle
        }


        # print(f'tshape_len = {targets_shape_len}, targets_shape = {targets_shape}')
        # print(f'dshape_len = {data_shape_len}, data_shape = {data_shape}')

        error = None

        try:
            # print(f"Python: received id={id}, model_offset={model_offset}, result_offset={result_offset}, model_size={model_size}, data_offset={data_offset}, data_shape={data_shape}, targets_offset={targets_offset}, targets_shape={targets_shape}")
            compute(
                mm_models, 
                mm_dataset, 
                mm_results,
                model_offset,
                result_offset,
                model_size,
                data_offset,
                data_shape,
                targets_offset,
                targets_shape,
                hyperparameters
            )

        except Exception as e:
            error = e
            # print error with traceback
            print(f"Python: error: {type(e)}: {e}")
            traceback.print_exc()

        resp_msg = struct.pack(OUT_FMT, id, 0 if not error else -1, model_offset, result_offset)
        mq_out.send(resp_msg, priority=0)
        # print(f"Python: sent id={id}, err_code={0 if not error else -1}")


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        if type(e).__name__ != "SignalError":
            print(f"Python: error: {type(e)}: {e}")

    cleanup()
    sys.exit(0)