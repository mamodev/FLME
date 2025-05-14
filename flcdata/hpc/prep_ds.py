import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.flcdata import Model, FLCDataset
import struct
import numpy as np

ds_path = "../.splits/data"
ds_name = os.path.basename(ds_path)

dss = FLCDataset.LoadGroups(ds_path, train=True)


buff = b''

buff += struct.pack('I', len(dss))
buff += struct.pack('I', dss[0].n_features)
buff += struct.pack('I', dss[0].n_classes)

for pidx, ds in enumerate(dss):
    data = ds.data  
    targets = ds.targets

    data_shape = data.shape
    targets_shape = targets.shape

    buff += struct.pack('I', len(data_shape))
    for dim in data_shape:
        buff += struct.pack('I', dim)

    buff += struct.pack('I', len(targets_shape))
    for dim in targets_shape:
        buff += struct.pack('I', dim)

for pidx, ds in enumerate(dss):
    data = ds.data.numpy()
    targets = ds.targets.numpy()

    datashape = data.shape
    targetsshape = targets.shape

    data = data.tobytes()
    targets = targets.tobytes()

    print(f"partition {pidx} data_starts={len(buff) + 4}  data_shape={datashape}")


    assert len(data) == np.prod(datashape) * 4
    assert len(targets) == np.prod(targetsshape) * 8

    buff += data

    print(f"partition {pidx} targets_starts={len(buff) + 4}  targets_shape={targetsshape}")
    buff += targets



print(len(buff))
buff = struct.pack('I', len(buff)) + buff

with open(f'{ds_name}.bin', 'wb') as f:
    f.write(buff)
    f.close()





