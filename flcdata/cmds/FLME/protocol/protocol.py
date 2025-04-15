from struct import pack, unpack, calcsize
from dataclasses import dataclass
from typing import Dict, Union, Tuple
from torch import Tensor, from_numpy
from io import BytesIO
import numpy as np
from datetime import datetime, timezone, timedelta

from immutabledict import immutabledict
import jwt
SECRET_KEY = "some_secret_key"

@dataclass  
class Auth:
    key: str
    cluster: int    
    gid: int
    pid: int
    exp: datetime = datetime.now(timezone.utc) + timedelta(hours=1)

    def encode(self) -> bytes:
        return jwt.encode({
            'key': self.key,
            'cluster': self.cluster,
            'gid': self.gid,
            'pid': self.pid,
            'exp': self.exp,
        }, SECRET_KEY, algorithm='HS256').encode('utf-8')
    
    def decode(token_buff: bytes) -> 'Auth':
        auth = jwt.decode(token_buff, SECRET_KEY, algorithms=["HS256"])
        return Auth(auth["key"], auth["cluster"], auth["gid"], auth["pid"], datetime.fromtimestamp(auth["exp"]))

    def __hash__(self):
        return hash(
            (
                self.key,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Auth):
            return NotImplemented
        
        return (
            self.key == other.key
        )
    
    def to_dict(self) -> Dict:
        return immutabledict({
            'key': self.key,
            'cluster': self.cluster,
            'gid': self.gid,
            'pid': self.pid,
            'exp': self.exp.timestamp(),
        })
    

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

def decode_client_token(token_buff: bytes) -> Union[Tuple[Dict, None], Tuple[Dict, bytes]]:
    try:
        auth = jwt.decode(token_buff, SECRET_KEY, algorithms=["HS256"])
        if "key" not in auth:
            raise jwt.InvalidTokenError("Token must contain a 'key' field")
        
        if "cluster" not in auth or type(auth["cluster"]) != int:   
            raise jwt.InvalidTokenError("Token must contain a 'cluster' field of type int")
        
        return Auth(auth["key"], auth["cluster"], auth["gid"], auth["pid"], datetime.fromtimestamp(auth["exp"])), None
    except jwt.ExpiredSignatureError:
        return None, create_response_packet(401, b"Token expired")
    except jwt.InvalidTokenError as e:
        return None, create_response_packet(401, str(e).encode())

def create_packet(id: int, payload: bytes | None) -> bytes:
    payload = payload if payload is not None else b''

    buff = pack('!H', id)
    buff += pack('!I', len(payload))
    buff += payload

    p = Packet.from_buffer(buff)
    assert p.id == id, f"Invalid serialization of packet, p.id !== id, {id} != {p.id}"
    assert len(p.buff) == len(payload), f"Invalid serialization of packet len(buff) != len(payload), {len(p.buff)} != {len(payload)}"
    assert p.buff == payload, "Invalid serialization of packet buff !== p.buff"

    return buff

def create_response_packet(code: int, payload: bytes) -> bytes:
    return create_packet(code, payload)

class Packet:
    id: int
    buff: bytes

    def __init__(self, id, buff: bytes):
        self.buff = buff
        self.id = id

    def from_buffer(buff: bytes):
        id = unpack("!H", buff[:2])[0]
        len = unpack("!I", buff[2:6])[0]
        return Packet(int(id), buff[6:len+6])

    def to_buffer(self) -> bytes:
        pass
        
AuthPacketID = 1
GetModelPacketID = 2
PutModelPacketID = 3
UpgrateToListener = 4

TrainEventID = 111
ForceSyncTrainEventID = 112
NewGlobalModelEventID = 113


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

    def __init__(self, version: int, model: ModelData):
        self.version = version
        self.model = model

    def from_buffer(buff: bytes) -> 'GetModelPacketResponse':
        assert len(buff) > 4, "buffer must have at least 4 bytes"
        version = unpack('!I', buff[:4])[0]
        model, bytes_read = ModelData.from_buffer(buff[4:])
        return GetModelPacketResponse(version, model)
    
    def to_buffer(self) -> bytes:
        buffer = pack('!I', self.version)
        buffer += self.model.to_buffer()
        return buffer

@dataclass
class ModelMeta:
    momentum: float
    learning_rate: float
    train_loss: float
    test_loss: float
    local_epoch: int
    train_samples: int

    def needed_bytes() -> int:
        return calcsize('!ddffii')

    def from_buffer(buff: bytes) -> 'ModelMeta':
        momentum, learning_rate, train_loss, test_loss, local_epoch, train_samples = unpack('!ddffii', buff)
        return ModelMeta(momentum, learning_rate, train_loss, test_loss, local_epoch, train_samples)
    
    def to_buffer(self) -> bytes:
        return pack('!ddffii', self.momentum, self.learning_rate, self.train_loss, self.test_loss, self.local_epoch, self.train_samples)
    
    def to_dict(self) -> Dict:
        return {
            "momentum": self.momentum,
            "learning_rate": self.learning_rate,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "local_epoch": self.local_epoch,
            "train_samples": self.train_samples,
        }

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
