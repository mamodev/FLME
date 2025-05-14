
def preprocess_dataset(path: str, out_path: str, verbose: bool = False):
    import os
    from preprocessors.flcdata import FLCDataset
    import struct
    import numpy as np

    assert os.path.exists(path), f"Path {path} does not exist"
    assert os.path.isdir(path), f"Path {path} is not a directory"

    assert type(out_path) == str and out_path != "", f"Output path {out_path} is not a valid string"

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    assert os.path.isdir(out_path), f"Output path {out_path} is not a directory"


    ds_name = os.path.basename(path)
    dss = FLCDataset.LoadGroups(path, train=True)
    
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

        if verbose:
            print(f"partition {pidx} data_starts={len(buff) + 4}  data_shape={datashape}")


        assert len(data) == np.prod(datashape) * 4
        assert len(targets) == np.prod(targetsshape) * 8

        buff += data
        if verbose:
            print(f"partition {pidx} targets_starts={len(buff) + 4}  targets_shape={targetsshape}")
        buff += targets


    if verbose:
        print(f"total size={len(buff)}")

    buff = struct.pack('I', len(buff)) + buff

    with open(os.path.join(out_path, ds_name + ".bin"), 'wb') as f:
        f.write(buff)
        f.close()

