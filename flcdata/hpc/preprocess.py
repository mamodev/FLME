import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.flcdata import Model, FLCDataset

ds = "../.splits/data"
in_size, out_size = FLCDataset.LoadSize(ds)
model = Model(insize=in_size, outsize=out_size)
dict = model.state_dict()

import io
import struct

buff = b''

nlayers = len(dict.keys())

buff += struct.pack('i', nlayers)

for key in dict.keys():
    print(key, dict[key].shape, dict[key].dtype)

    ascii_key = key.encode('ascii')
    key_len = len(ascii_key)
    buff += struct.pack('i', key_len)
    buff += ascii_key

    shape_len = len(dict[key].shape)
    buff += struct.pack('i', shape_len)

    for dim in dict[key].shape:
        buff += struct.pack('i', dim)

for key in dict.keys():
    buff += dict[key].numpy().tobytes() 


print(len(buff))
buff = struct.pack('i', len(buff)) + buff


with open('model.bin', 'wb') as f:
    f.write(buff)
    f.close()
