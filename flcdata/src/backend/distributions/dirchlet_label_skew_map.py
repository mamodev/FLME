from backend.interfaces import Module, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter



def dirchlet_label_skew_map(n_classes, n_samples, n_partitions, beta=1.0, alpha=1.0, seed=None):
    import numpy as np

    n_samples_per_class = n_samples // n_classes
    n_samples_per_class_remainder = n_samples % n_classes
    n_samples_per_class = [
        n_samples_per_class + (1 if i < n_samples_per_class_remainder else 0)
        for i in range(n_classes)
    ]
    
    np.random.seed(seed)
    knowledge = np.random.dirichlet([beta] * n_partitions)
    knowledge = [int(x * n_samples) for x in knowledge]
    __knowledge_rem = n_samples - sum(knowledge)
    if __knowledge_rem > 0:
        np.random.seed(seed)
        rndm_idx = np.random.randint(0, n_partitions)
        knowledge[rndm_idx] += n_samples - sum(knowledge)

    dir_weights = np.zeros((n_classes, n_partitions), dtype=int)

    for part in range(n_partitions):
        np.random.seed(seed)
        _class_distr = np.random.dirichlet([alpha] * n_classes)
        partition_samples = knowledge[part]

        class_distr = [int(x * partition_samples) for x in _class_distr]
        __class_distr_rem = partition_samples - sum(class_distr)
        if __class_distr_rem > 0:
            np.random.seed(seed)
            rndm_idx = np.random.randint(0, n_classes)
            class_distr[rndm_idx] += partition_samples - sum(class_distr)
            
        assert sum(class_distr) == partition_samples, f"something wrong with the distribution, sum should be {partition_samples} but is {sum(class_distr)}"

        classes_iter = list(range(n_classes))
        # np.random.shuffle(classes_iter)

        not_assigned = 0
        for c in classes_iter:
            assigned_samples = sum(dir_weights[c, :])
            remaining_samples = n_samples_per_class[c] - assigned_samples
            assert remaining_samples >= 0

            if remaining_samples >= class_distr[c]:
                dir_weights[c, part] = class_distr[c]
            else:
                dir_weights[c, part] = remaining_samples
                to_distribute = class_distr[c] - remaining_samples
                not_assigned += to_distribute

        assert type(not_assigned) in [
            int, np.int32, np.int64
        ], f"something wrong with the distribution, not_assigned should be int but is {type(not_assigned)}"

        i = 0
        to_distribute = not_assigned
        while to_distribute > 0:
            # free in form of (class, free, original_prc)[]
            free = [(j, n_samples_per_class[j] - sum(dir_weights[j, :]), _class_distr[j]) for j in range(n_classes)]
            
            # filter out classes that are already full
            free = [p for p in free if p[1] > 0]

            # previus probability distribution summed to 1 so we should recalculate partition proportions
            _psum = sum([p[2] for p in free])
            free = [(c, int(f), p / _psum) for c, f, p in free]

            # now we just check to be sure that the sum of the probabilities is 1
            psum = sum([p[2] for p in free])
            assert np.round(psum) == 1, f"something wrong with the distribution, psum should be 1 but is {np.round(psum)} previous psum was {_psum}, to_distribute = {to_distribute}, free = {free}"
            
            cd = [(c, f, int(p * to_distribute)) for c, f, p in free]
            cd = [(c, f, n) for c, f, n in cd if n > 0]
            if len(cd) == 0:
                # peack a random class that has enough space
                np.random.seed(seed)
                c = np.random.choice([c for c, f, n in free if f >= to_distribute])
                cd = [(c, to_distribute, to_distribute)]

            for c, f, n in cd:
                to_distribute -= min(n, f)
                dir_weights[c, part] += min(n, f)

            i += 1
            assert i < 10, f"Something wrong with the distribution, i = {i} > 5 and to_distribute = {to_distribute}"

    return dir_weights

def generate(params, n_classes, n_samples):

    return dirchlet_label_skew_map(
        n_classes=n_classes,
        n_samples=n_samples,
        n_partitions=params["n_partitions"],
        beta=params["beta"],
        alpha=params["alpha"],
        seed=params["seed"],
    )

generator = Module(
    name="dirchlet_label_skew_map",
    description="",
    parameters={
        "n_partitions": IntParameter(1, 10000, 10),
        "beta": FloatParameter(0.0, 1.0, 0.5),
        "alpha": FloatParameter(0.0, 1.0, 0.5),
        "seed": IntParameter(1, 1000000000, 1),
    },
    fn=generate,
)    
