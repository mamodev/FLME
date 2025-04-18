from backend.interfaces import Module, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter
from typing import List, Tuple
import numpy as np

SubspaceCoord = Tuple[float, float, float, float, float, float]
Vec3 = Tuple[float, float, float]
def calc_subspaces(space_size, separations, n_labels) -> List[SubspaceCoord]:
    '''
    Calculate the subspaces for each label
    The subspace are rapresented by the coordinates of the two opposite corners of the cuboid
    example: (0, 0, 0, 1, 1, 1) is a subspace that goes from (0, 0, 0) to (1, 1, 1)
    '''

    one_split_every = (space_size * 2) / (separations + 1)
    subspaces = []
    ss = space_size
    for i in range(separations + 1):
        for j in range(separations + 1):
            for k in range(separations + 1):
                startX = -ss + i * one_split_every
                endX = -ss + (i + 1) * one_split_every
                startY = -ss + j * one_split_every
                endY = -ss + (j + 1) * one_split_every
                startZ = -ss + k * one_split_every
                endZ = -ss + (k + 1) * one_split_every
                subspaces.append((startX, startY, startZ, endX, endY, endZ))

    assert len(subspaces) == n_labels, f"Expected {n_labels} subspaces, got {len(subspaces)}"
    return subspaces


def random_subspace_point(subspace: SubspaceCoord) -> Vec3:
    sx, sy, sz, ex, ey, ez = subspace
    return (np.random.uniform(sx, ex), np.random.uniform(sy, ey), np.random.uniform(sz, ez))

def random_subspace_points(subspace: SubspaceCoord, n: int) -> List[Vec3]:
    return [
        random_subspace_point(subspace)
        for _ in range(n)
    ]

def generate(params):
    n_separations = params["separations"]
    n_labels = (n_separations + 1) ** 3
    n_samples = params["n_samples"]
    seed = params["seed"]


    np.random.seed(seed)
    subspaces = calc_subspaces(
        space_size=1,
        separations=n_separations,
        n_labels=n_labels
    )

    XX = []
    YY = []
    for i, ss in enumerate(subspaces):
        samples = random_subspace_points(ss, n_samples // n_labels)
        XX.extend(samples)
        YY.extend([i] * len(samples))
    
    XX = np.array(XX)
    YY = np.array(YY)
    assert len(XX) == len(YY), f"XX and YY have different lengths: {len(XX)} != {len(YY)}"
    assert len(XX) == n_samples, f"XX has different length than n_samples: {len(XX)} != {n_samples}"

    # random shuffle
    indices = np.random.permutation(len(XX))
    XX = XX[indices]
    YY = YY[indices]



    return XX, YY


generator = Module(
    name="cuboids",
    description="cuboids generator",
    parameters={
        "n_samples": IntParameter(1, 10000, 2000),
        "separations": IntParameter(1, 100, 1),
        "seed": IntParameter(1, 1000000000, 1),
    },
    fn=generate,
)

