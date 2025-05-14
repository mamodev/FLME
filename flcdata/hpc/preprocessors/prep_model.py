def preprocess_model(ds_path: str, out_path: str, verbose: bool = False):
    from preprocessors.flcdata import Model, FLCDataset
    import struct
    import os

  
    assert type(ds_path) == str and ds_path != "", f"Path {ds_path} is not a valid string"
    assert os.path.exists(ds_path), f"Path {ds_path} does not exist"
    assert os.path.isdir(ds_path), f"Path {ds_path} is not a directory"

    assert type(out_path) == str and out_path != "", f"Output path {out_path} is not a valid string"
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    assert os.path.exists(out_path), f"Output path {out_path} does not exist"
    assert os.path.isdir(out_path), f"Output path {out_path} is not a directory"

    in_size, out_size = FLCDataset.LoadSize(ds_path)
    model = Model(insize=in_size, outsize=out_size)
    dict = model.state_dict()

    buff = b''
    nlayers = len(dict.keys())
    buff += struct.pack('i', nlayers)

    for key in dict.keys():
        if verbose:
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


    if verbose:
        print("model size: ", len(buff))
    
    buff = struct.pack('i', len(buff)) + buff

    with open(os.path.join(out_path, "model.bin"), 'wb') as f:
        f.write(buff)
        f.close()
