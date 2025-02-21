from struct import pack, unpack, calcsize
from dataclasses import dataclass
from typing import Dict, Union
from torch import Tensor, from_numpy
from io import BytesIO
import numpy as np
import jwt
from datetime import datetime, timezone, timedelta

SECRET_KEY = "some_secret_key"

def get_client_token(gid: int, pid: int) -> bytes:
    utc_now = datetime.now(timezone.utc)
    jwt_token = jwt.encode({
        'key': f'{gid}-{pid}',
        "cluster": 1,
        'gid': gid,
        'pid': pid,
        'exp': utc_now + timedelta(hours=1),
    }, SECRET_KEY, algorithm='HS256')

    return jwt_token.encode('utf-8')

def create_packet(id: int, payload: bytes) -> bytes:
    buff = pack('!H', id)
    buff += pack('!I', len(payload))
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

    def from_buffer(buff: bytes):
        buffer_len = unpack('!I', buff[:4])[0]
        buff = buff[4:4 + buffer_len]

        bytes_read = buffer_len + 4

        state_buffer = BytesIO(buff)
        np_state_dict = np.load(state_buffer, allow_pickle=True).item()
        state_dict = {}
        for k, v in np_state_dict.items():
            state_dict[k] = from_numpy(v)

        return ModelData(state_dict), bytes_read
    
    def to_buffer(self) -> bytes:
        buffer = BytesIO()
        np.save(buffer, {k: v.numpy() for k, v in self.model_state.items()})
        buffer = buffer.getvalue()
        buff = pack("!I", len(buffer))
        return buff + buffer


class GetModelPacket:
    model_version: int

    def __init__(self, model_version: int):
        self.model_version = model_version

    def from_buffer(buff: bytes) -> 'GetModelPacket':
        model_version = unpack('!I', buff)[0]
        return GetModelPacket(model_version)
    
    def to_buffer(self) -> bytes:
        return pack('!I', self.model_version)

class GetModelPacketResponse:
    version: int
    model: ModelData
    split_model_data_version: int
    split_model_data: ModelData

    def __init__(self, version: int, model: ModelData, split_model_data_version: int = None, split_model_data: ModelData = None):
        self.version = version
        self.model = model
        self.split_model_data = split_model_data
        self.split_model_data_version = split_model_data_version

    def from_buffer(buff: bytes) -> 'GetModelPacketResponse':
        version = unpack('!I', buff[:4])[0]
        model, bytes_read = ModelData.from_buffer(buff[4:])
        offset = 4 + bytes_read
        has_split_model_data = unpack('!?', buff[offset:offset + 1])[0]
        offset += 1
        if has_split_model_data:
            split_model_data_version = unpack('!I', buff[offset:offset + 4])[0]
            offset += 4
            split_model_data, split_bytes_read = ModelData.from_buffer(buff[offset:])
            offset += split_bytes_read
        else:
            split_model_data_version = 0
            split_model_data = None
        return GetModelPacketResponse(version, model, split_model_data_version, split_model_data)
    
    def to_buffer(self) -> bytes:
        buffer = pack('!I', self.version)
        buffer += self.model.to_buffer()
        has_split_model_data = self.split_model_data is not None
        buffer += pack('!?', has_split_model_data)
        if has_split_model_data:
            buffer += pack('!I', self.split_model_data_version)
            buffer += self.split_model_data.to_buffer()
        return buffer


@dataclass
class ModelMeta:
    momentum: float
    learning_rate: float
    train_loss: float
    test_loss: float
    local_epoch: int
    train_samples: int
    derived_from: int = 0

    def needed_bytes() -> int:
        return calcsize('!ddffii')

    def from_buffer(buff: bytes) -> 'ModelMeta':
        momentum, learning_rate, train_loss, test_loss, local_epoch, train_samples = unpack('!ddffii', buff)
        return ModelMeta(momentum, learning_rate, train_loss, test_loss, local_epoch, train_samples)
    
    def to_buffer(self) -> bytes:
        return pack('!ddffii', self.momentum, self.learning_rate, self.train_loss, self.test_loss, self.local_epoch, self.train_samples)


class PutModelPacket:
    meta: ModelMeta
    data: ModelData
    
    def __init__(self, data: Union[Dict[str, Tensor], ModelMeta], meta: ModelMeta):
        self.data = data if isinstance(data, ModelData) else ModelData(data)
        self.meta = meta

    def from_buffer(buff: bytes) -> 'PutModelPacket':
        
        meta = ModelMeta.from_buffer(buff[:ModelMeta.needed_bytes()])
        data, _ = ModelData.from_buffer(buff[ModelMeta.needed_bytes():])
        return PutModelPacket(data, meta)
        
    def to_buffer(self) -> bytes:
        return self.meta.to_buffer() + self.data.to_buffer()
    

# CLIENT UTILS
import socket
from typing import Tuple
from torch_utils import deepCloneToCpu

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
    packet_len = unpack('!I', packet_len)[0]
    payload = recvAll(s, packet_len)

    return int.from_bytes(packet_id, 'big'), payload

def connect_with_auth(host: str, port: int, gid: int, pid: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(create_packet(AuthPacketID, get_client_token(gid, pid)))
    packet_id, payload = recv_packet(s)
    if packet_id != 0:
        raise Exception(f"Failed to authenticate: {payload}")
    return s


def get_latest_model(s: socket.socket) -> GetModelPacketResponse:
    s.sendall(create_packet(GetModelPacketID, GetModelPacket(0).to_buffer()))
    packet_id, payload = recv_packet(s)
    if packet_id != 0:
        raise Exception(f"Failed to get model: {payload}")
    
    res = GetModelPacketResponse.from_buffer(payload)
    return res

def put_model(s: socket.socket, state: Dict[str, Tensor], meta: ModelMeta) -> None:
    putModelPacket = PutModelPacket(ModelData(deepCloneToCpu(state)), meta)
    putModelPacketBuffer = putModelPacket.to_buffer()

    s.sendall(create_packet(PutModelPacketID, putModelPacketBuffer))
    packet_id, payload = recv_packet(s)
    if packet_id != 0:
        raise Exception(f"Failed to put model: {payload}")