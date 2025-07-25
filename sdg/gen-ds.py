from flcdata import Model, FLCDataset

ds_path = ".data/cubeds-drift"

insize, outsize = FLCDataset.LoadSize(ds_path)

print(f"Input size: {insize}, Output size: {outsize}")

dss = FLCDataset.LoadGroupsAndLinearPartition(
    ds_path,
    10,
    shuffle=False,
)

labels = set([
    ds.dex
    for ds in dss
])

labels = [i for i in labels ]





