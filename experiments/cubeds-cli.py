
from cubeds import CuboidDatasetMeta
import argparse

import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
import plot_utils
from plot_utils import MRK_PLT, CLR_PLT

import os

SEPARATIONS = 1
Vec3 = Tuple[float, float, float]


SHOW_SUBSPACES = False
SHOW_CENTERS = False

    # python cubeds-cli.py --nsamples 20000 --ngroups 8 --void 4 --space-size=10 --name lswap-skew --lswap 0,0,1 --lswap 1,1,2 --lswap 2,2,3 --lswap 3,3,4 --lswap 4,4,5 --lswap 5,5,6 --lswap 6,6,7 --lswap 7,7,0
    # python cubeds-cli.py --nsamples 20000 --ngroups 8 --void 4 --space-size=10 --name more-skew --ilswap 0,0,1 --ilswap 1,1,2 --ilswap 2,2,3 --ilswap 3,3,4 --ilswap 4,4,5 --ilswap 5,5,6 --ilswap 6,6,7 --ilswap 7,7,0 --lswap 7,0,1 --lswap 6,1,2 --lswap 5,2,3 --lswap 4,3,4 --lswap 3,4,5 --lswap 2,5,6 --lswap 1,6,7 --lswap 0,7,0
def parse_args():
    global SHOW_SUBSPACES, SHOW_CENTERS

    parser = argparse.ArgumentParser(description='Create a synthetic dataset')

    parser.add_argument("--name", dest="ds_name", type=str, default="cuboids", help="Name of the dataset")
    parser.add_argument('--ngroups', type=int, default=4, help='Number of clients')
    parser.add_argument('--nsamples', type=int, default=500, help='Number of points per label')
    parser.add_argument('--gui', action='store_true', help='Show the GUI', default=False)
    parser.add_argument('--testprc', type=float, default=0.2, help='How many points to use for testing')
    parser.add_argument('--void', type=int, default=0, help='How many points to use for testing')
    parser.add_argument('--space-size', type=int, default=1, help='Size of the space')
    parser.add_argument('--plot-backend', type=str, default='GTK3Agg', help='Plot backend')
    parser.add_argument('--root', type=str, default='./data', help='Root folder for the dataset')
    
    parser.add_argument('--lswap', action='append', help='Usage: --lswap <gid>,<lid1>,<lid2>, for <gid> set <lid1> to <lid2>')
    parser.add_argument('--ilswap', action='append', help='Usage: --ilswap <gid>,<lid1>,<lid2>, for a group <gid> set <lid1> to <lid2> and viceversa')
    parser.add_argument('--no-fskew', action='store_true', help='Do not skew the features', default=False)

    parser.add_argument('--show-subspaces', action='store_true', help='Show the subspaces', default=False)
    parser.add_argument('--show-centers', action='store_true', help='Show the centers', default=False)

    args = parser.parse_args()

    SHOW_SUBSPACES = args.show_subspaces
    SHOW_CENTERS = args.show_centers

    n_labels = (SEPARATIONS + 1)**3 

    between = lambda x, a, b: a <= x <= b

    lswap = [] if args.lswap is None else args.lswap
    ilswap = [] if args.ilswap is None else args.ilswap


    def assert_swap(s):
        assert len(s) == 3, f"Swap {s} must have 3 elements"
        assert between(s[0], 0, args.ngroups - 1), f"Group id {s[0]} out of range"
        assert between(s[1], 0, n_labels - 1), f"Label id {s[1]} out of range"
        assert between(s[2], 0, n_labels - 1), f"Label id {s[2]} out of range"

    for i, l in enumerate(lswap):
        lswap[i] = [int(x) for x in l.split(",")]
        assert_swap(lswap[i])

    for i, l in enumerate(ilswap):
        ilswap[i] = [int(x) for x in l.split(",")]
        assert_swap(ilswap[i])

    swap = {gid: [] for gid in range(args.ngroups)}
    for gid, l1, l2 in lswap:
        swap[gid].append((l1, l2))

    for gid, l1, l2 in ilswap:
        swap[gid].append((l1, l2))
        swap[gid].append((l2, l1))

    meta = CuboidDatasetMeta(   
        n_groups=args.ngroups,
        n_labels=n_labels,
        n_void_groups=args.void,
        space_size=args.space_size,
        n_samples_per_label=args.nsamples,
        test_prc=args.testprc,
        groups_centers=[],
        lable_swaps=swap,
        features_skew=not args.no_fskew
    )

    if args.gui:
        import matplotlib
        matplotlib.use(args.plot_backend)

    return meta, args.gui, args.root, args.ds_name if args.ds_name else meta.get_name()


def init_gui(gui: bool, meta: CuboidDatasetMeta):
    if not gui:
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-meta.space_size, meta.space_size)
    ax.set_ylim(-meta.space_size, meta.space_size)
    ax.set_zlim(-meta.space_size, meta.space_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show(block=False)
    return ax


SubspaceCoord = Tuple[float, float, float, float, float, float]
def calc_subspaces(meta: CuboidDatasetMeta, ax) -> List[SubspaceCoord]:
    '''
    Calculate the subspaces for each label
    The subspace are rapresented by the coordinates of the two opposite corners of the cuboid
    example: (0, 0, 0, 1, 1, 1) is a subspace that goes from (0, 0, 0) to (1, 1, 1)
    '''

    one_split_every = (meta.space_size * 2) / (SEPARATIONS + 1)
    subspaces = []
    ss = meta.space_size
    for i in range(SEPARATIONS + 1):
        for j in range(SEPARATIONS + 1):
            for k in range(SEPARATIONS + 1):
                startX = -ss + i * one_split_every
                endX = -ss + (i + 1) * one_split_every
                startY = -ss + j * one_split_every
                endY = -ss + (j + 1) * one_split_every
                startZ = -ss + k * one_split_every
                endZ = -ss + (k + 1) * one_split_every
                subspaces.append((startX, startY, startZ, endX, endY, endZ))

                if ax and SHOW_SUBSPACES:
                    plot_utils.plot_cube_by_vertex(ax, (startX, startY, startZ), (endX, endY, endZ), "white", alpha=0.1)

    assert len(subspaces) == meta.n_labels, f"Expected {meta.n_labels} subspaces, got {len(subspaces)}"
    return subspaces



def random_subspace_point(subspace: SubspaceCoord) -> Vec3:
    sx, sy, sz, ex, ey, ez = subspace
    return (np.random.uniform(sx, ex), np.random.uniform(sy, ey), np.random.uniform(sz, ez))

def random_subspace_points(subspace: SubspaceCoord, n: int) -> List[Vec3]:
    return [
        random_subspace_point(subspace)
        for _ in range(n)
    ]

def subspace_center(subspace: SubspaceCoord) -> Vec3:
    sx, sy, sz, ex, ey, ez = subspace
    return ((sx + ex) / 2, (sy + ey) / 2, (sz + ez) / 2)

def partition_points(points: List[Vec3], centers: List[Vec3]) -> List[List[Vec3]]:
    points = np.array(points)
    centers = np.array(centers)
    
    # this is an array of shape (n_points, n_centers)
    # where each element is the distance of the point from the center
    distances = np.array([
        np.linalg.norm(points - center, axis=1)
        for center in centers
    ]) 

    closest_centers = np.argmin(distances, axis=0)

    return [points[closest_centers == i] for i in range(len(centers))]

def partition_in_groups(subspace_points: list[list[Vec3]], gcenters: list[list[Vec3]]):
    groups_points : list[list[Vec3]] = [list() for _ in range(len(gcenters))]

    for i, sub_points in enumerate(subspace_points):
        sub_centers = [s[i] for s in gcenters]
        sub_points_groups = partition_points(sub_points, sub_centers)

        for j, points in enumerate(sub_points_groups):
            groups_points[j].append(points)

    return groups_points

def random_partition(subspace_points: list[list[Vec3]], npartitions: int):
    partitioned_points: list[list[Vec3]] = [list() for _ in range(npartitions)]

    for ss_points in subspace_points:
        np.random.shuffle(ss_points)
        partition_size = len(ss_points) // npartitions

        for i in range(npartitions):
            start = i * partition_size
            end = (i + 1) * partition_size
            partitioned_points[i].append(ss_points[start:end])

    return partitioned_points

def creation_cli():
    meta, gui, root, dsname = parse_args()
    ax = init_gui(gui, meta)
    novoid = lambda l: l if meta.n_void_groups == 0 else l[:-meta.n_void_groups]

    def get_group_ss_label(gidx, ss_idx) -> int:
        for i, (g, ss) in enumerate(meta.lable_swaps[gidx]):
            if ss == ss_idx:
                return g

        return ss_idx

    subspaces = calc_subspaces(meta, ax)
    subspaces_points = [
        random_subspace_points(ss, meta.n_samples_per_label) for ss in subspaces
    ]

    total_groups = meta.n_groups + meta.n_void_groups

    if meta.features_skew:
        groups_subspaces_centers = [
            [random_subspace_point(ss) for ss in subspaces] for _ in range(total_groups) 
        ]
    else:
        groups_subspaces_centers = [
            [subspace_center(ss) for ss in subspaces] for _ in range(total_groups)
        ]

    if gui and SHOW_CENTERS:
        for gidx, xx in enumerate(novoid(groups_subspaces_centers)):
            for ssidx, x in enumerate(xx):
                lidx = get_group_ss_label(gidx, ssidx)
                ax.scatter(x[0], x[1], x[2], s=200, marker=MRK_PLT[lidx], c=CLR_PLT[gidx])

    if meta.features_skew:
        groups_points = partition_in_groups(subspaces_points, groups_subspaces_centers)
    else:
        groups_points = random_partition(subspaces_points, total_groups)

    if gui:
        for gidx, xx in enumerate(novoid(groups_points)):
            for ss_idx, x in enumerate(xx):
                x = np.array(x)
                lidx = get_group_ss_label(gidx, ss_idx)
                ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=40, alpha=0.5, marker=MRK_PLT[lidx], c=CLR_PLT[gidx])

    if gui:
        plt.show()
        return  


    # save dataset_to_folder !
    root_ds_folder = f"{root}/cuboids"
    if not os.path.exists(root_ds_folder):
        os.mkdir(root_ds_folder)

    curr_ds_folder = f"{root_ds_folder}/{dsname}"

    if os.path.exists(curr_ds_folder):
        res = input(f"Dataset {curr_ds_folder} already exists. Do you want to overwrite it? (y/n)")
        if res != 'y':
            print("Dataset creation aborted")
            return
        
    os.system(f"rm -rf {curr_ds_folder}")
    os.mkdir(curr_ds_folder)
    os.mkdir(f"{curr_ds_folder}/train")
    os.mkdir(f"{curr_ds_folder}/test")

    for gidx, ss_points in enumerate(novoid(groups_points)):
        for ss_idx, xx in enumerate(ss_points):
            n_test = int(meta.test_prc * len(xx))

            test_points = xx[:n_test]
            train_points = xx[n_test:]

            np.save(f"{curr_ds_folder}/test/g{gidx}_l{ss_idx}.npy", test_points)
            np.save(f"{curr_ds_folder}/train/g{gidx}_l{ss_idx}.npy", train_points)

    meta.save(f"{curr_ds_folder}/META.json")

    print(f"Dataset {curr_ds_folder} created.")

if __name__ == "__main__":
    creation_cli()