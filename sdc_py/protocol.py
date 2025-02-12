import struct
import dataclasses  

from typing import Dict, Tuple, Union
from torch import Tensor, from_numpy
import io
import numpy as np

def create_packet(id: int, payload: bytes) -> bytes:
    buff = struct.pack('!H', id)
    buff += struct.pack('!I', len(payload))
    buff += payload
    return buff



def create_response_packet(code: int, payload: bytes) -> bytes:
    return create_packet(code, payload)



class Packet:
    def from_buffer(buff: bytes):
        pass

    def to_buffer(self) -> bytes:
        pass
        
AuthPacketID = 1
GetModelPacketID = 2
PutModelPacketID = 3


class ModelData:
    model_state: Dict[str, Tensor]

    def __init__(self, model_state: Dict[str, Tensor]):
        self.model_state = model_state

    def from_buffer(buff: bytes) -> 'ModelData':
        state_buffer = io.BytesIO(buff)
        np_state_dict = np.load(state_buffer, allow_pickle=True).item()
        state_dict = {}
        for k, v in np_state_dict.items():
            state_dict[k] = from_numpy(v)

        return ModelData(state_dict)
    
    def to_buffer(self) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, {k: v.numpy() for k, v in self.model_state.items()})
        return buffer.getvalue()


class GetModelPacket:
    model_version: int

    def __init__(self, model_version: int):
        self.model_version = model_version

    def from_buffer(buff: bytes) -> 'GetModelPacket':
        model_version = struct.unpack('!I', buff)[0]
        return GetModelPacket(model_version)
    
    def to_buffer(self) -> bytes:
        return struct.pack('!I', self.model_version)
    

@dataclasses.dataclass
class ModelMeta:
    momentum: float
    learning_rate: float
    train_loss: float
    test_loss: float
    local_epoch: int
    train_samples: int
    derived_from: int = 0

    def needed_bytes() -> int:
        return struct.calcsize('!ddffii')

    def from_buffer(buff: bytes) -> 'ModelMeta':
        momentum, learning_rate, train_loss, test_loss, local_epoch, train_samples = struct.unpack('!ddffii', buff)
        return ModelMeta(momentum, learning_rate, train_loss, test_loss, local_epoch, train_samples)
    
    def to_buffer(self) -> bytes:
        return struct.pack('!ddffii', self.momentum, self.learning_rate, self.train_loss, self.test_loss, self.local_epoch, self.train_samples)


class PutModelPacket:
    meta: ModelMeta
    data: ModelData
    
    def __init__(self, data: Union[Dict[str, Tensor], ModelMeta], meta: ModelMeta):
        self.data = data if isinstance(data, ModelData) else ModelData(data)
        self.meta = meta

    def from_buffer(buff: bytes) -> 'PutModelPacket':
        
        meta = ModelMeta.from_buffer(buff[:ModelMeta.needed_bytes()])
        data = ModelData.from_buffer(buff[ModelMeta.needed_bytes():])
        return PutModelPacket(data, meta)
        
    def to_buffer(self) -> bytes:
        return self.meta.to_buffer() + self.data.to_buffer()
