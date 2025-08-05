from flcdata import FLCDataset, SimpleBatchLoader, VerySimpleModel
import torch
from torch.nn import functional as F
    
def deep_clone_state_dict(state_dict):
    return {k: v.clone() for k, v in state_dict.items()}

ds_path = ".data/cubeds-drift"

insize, outsize = FLCDataset.LoadSize(ds_path)

print(f"Input size: {insize}, Output size: {outsize}")

dss = FLCDataset.LoadGroupsAndLinearPartition(
    ds_path,
    10,
    shuffle=False,
)

test_dss = FLCDataset.LoadGroupsAndLinearPartition(
    ds_path,
    10,
    shuffle=False,
    train=False,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(len(dss)):
    dss[i].to_device(device)
    test_dss[i].to_device(device)


loaders = [
    SimpleBatchLoader(ds.data, ds.targets, batch_size=2048)
    for ds in dss
]

test_loaders = [
    SimpleBatchLoader(ds.data, ds.targets, batch_size=2048)
    for ds in test_dss
]



print(f"Total number of clients after partitioning: {len(dss)}")



rounds = 1000 
lr = 0.1
ephocs = 10
momentum = 0.9

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_ylim(0, 1)
plt.show(block=False)
plt.pause(0.1)



class WndMetric:
    def __init__(self, size):
        self.size = size
        self.data = []

    def add(self, value):
        self.data.append(value)
        if len(self.data) > self.size:
            self.data.pop(0)

    def avg(self):
        return sum(self.data) / len(self.data) if self.data else 0

    def var(self):
        avg = self.avg()
        return sum((x - avg) ** 2 for x in self.data) / len(self.data) if self.data else 0
    
    def trend(self):
        if len(self.data) < 2:
            return 0
        trend = 0
        for i in range(1, len(self.data)):
            perc_imp = (self.data[i] - self.data[i-1]) / self.data[i-1]
            trend += perc_imp
        return trend / (len(self.data) - 1)

wacc = WndMetric(10)
wloss = WndMetric(10)
wacc_var = WndMetric(10)
wloss_var = WndMetric(10)


Model = VerySimpleModel
nets = [
    Model(insize=insize, outsize=outsize).to(device)
    for _ in range(len(dss))
]

client_cluster = [
    0
    for _ in range(len(dss))
]

n_clusters = 1

start_models = [
    nets[0].state_dict()
]

for i in range(1, len(nets)):
    nets[i].load_state_dict(deep_clone_state_dict(start_models[client_cluster[i]]))

class WndStateDictDir:
    def __init__(self, size):
        self.size = size
        self.data = []

    def add(self, value, ref):
        assert isinstance(value, dict) and isinstance(ref, dict)
        assert len(value) == len(ref)
        assert all(k in ref for k in value)

        self.data.append({k: (value[k] - ref[k]).clone() for k in value})
        if len(self.data) > self.size:
            self.data.pop(0)

    def avg(self):
        if not self.data:
            return None

        avg = {}
        for k in self.data[0].keys():
            avg[k] = torch.zeros_like(self.data[0][k])
            for d in self.data:
                avg[k] += d[k]
            avg[k] /= len(self.data)
        return avg
    

wnd_dirs = [
    WndStateDictDir(10)
    for _ in range(len(dss))
]

accuracies = [[] for _ in range(len(dss))]

for iters in range(10):
    for round in range(rounds):
        lr = 0.01

        models = []
        for i, ds in enumerate(dss):
            model = nets[i]
            model.train()
            model.load_state_dict(deep_clone_state_dict(start_models[client_cluster[i]]))
        
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=1e-4,
            )

            running_loss = 0.0
            num_batches = 0
            for epoch in range(ephocs):
                for x, y in loaders[i]:
                    optimizer.zero_grad()
                    output = model(x)
                    loss = F.nll_loss(output, y)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_batches += 1

            avg_loss = running_loss / num_batches

            wnd_dirs[i].add(model.state_dict(), start_models[client_cluster[i]])
        
            models.append((model.state_dict(), ds.n_samples, avg_loss))


        test_accs = []
        oracle_accs = []

        # Aggregate the models using FedAvg
        for c in range(n_clusters):

            new_model = {}
            __models = [m for i, m in enumerate(models) if client_cluster[i] == c]

            for k in __models[0][0].keys():
                new_model[k] = torch.zeros_like(__models[0][0][k])
                for model, n_samples, _ in __models:
                    new_model[k] += model[k] * (n_samples / sum([m[1] for m in __models]))

            start_models[c] = new_model

            for i, ds in enumerate(test_dss):
                if client_cluster[i] != c:
                    continue

                model = nets[i]
                model.load_state_dict(deep_clone_state_dict(new_model))
                model.eval()
            
                total = 0
                correct = 0
                model.eval()
                with torch.no_grad():
                    for x, y in test_loaders[i]:
                        output = model(x)
                        _, predicted = torch.max(output.data, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()
                
                test_acc = 100 * correct / total
                test_accs.append(test_acc)
                accuracies[i].append(test_acc / 100)



            # net = Model(insize=insize, outsize=outsize).to(device)
            # net.load_state_dict(new_model)
            # total = 0
            # correct = 0
            # for i, ds in enumerate(test_dss):
            #     net.eval()
            #     with torch.no_grad():
            #         for x, y in test_loaders[i]:
            #             output = net(x)
            #             _, predicted = torch.max(output.data, 1)
            #             total += y.size(0)
            #             correct += (predicted == y).sum().item()

            # oracle_acc = 100 * correct / total
            oracle_acc = 0
            oracle_accs.append(oracle_acc)
    
        oracle_acc = sum(oracle_accs) / len(oracle_accs)
        avg_test = sum(test_accs) / len(test_accs)
        avg_loss = sum([m[2] for m in models]) / len(models)

        var_test = sum([(x - avg_test) ** 2 for x in test_accs]) / len(test_accs)
        var_loss = sum([(x - avg_loss) ** 2 for x in [m[2] for m in models]]) / len(models)

        wacc.add(avg_test)
        wloss.add(avg_loss)
        wacc_var.add(var_test)
        wloss_var.add(var_loss)


        def fmt_prc(x):
            RED_ANSI = "\033[91m"
            GREEN_ANSI = "\033[92m"
            RESET_ANSI = "\033[0m"
            formatted_number = f"{(x * 100):6.2f}"
        
            if x > 0:
                return f"{GREEN_ANSI}{formatted_number}{RESET_ANSI}"
            elif x < 0:
                return f"{RED_ANSI}{formatted_number}{RESET_ANSI}"
            
            return formatted_number
        
        print(
            f"R{round + 1:02d}/{rounds} | "
            f"Acc: {avg_test:.2f} {wacc.avg():.2f} {fmt_prc(wacc.trend())}, {var_test:.2f} {wacc_var.avg():.2f} {fmt_prc(wacc_var.trend())} | "
            f"Loss: {avg_loss:.2f} {wloss.avg():.2f} {fmt_prc(wloss.trend())}, {var_loss:.2f} {wloss_var.avg():.2f} {fmt_prc(wloss_var.trend())} | "
            f"Oracle Acc: {oracle_acc:.2f}% | "
        )  
    

        # assert len(test_accs) == len(dss), f"Expected {len(dss)} test accuracies, got {len(test_accs)}"

        # for c in range(n_clusters):
        #     n_clients_prev = sum([1 for i in range(len(dss)) if client_cluster[i] < c])

        #     for i in range(len(dss)):
        #         if client_cluster[i] == c:
        #             cluster_idx = sum([1 for j in range(len(dss)) if client_cluster[j] == c and j < n_clients_prev])
        #             _acc = test_accs[n_clients_prev + cluster_idx]
        #             accuracies[i].append(_acc/100)

        ax.clear()
        ax.set_ylim(0, 1)
        for i, acc in enumerate(accuracies):
            ax.plot(range(len(acc)), acc, label=f"Client {i+1}", alpha=0.5)


        plt.draw()
        plt.pause(0.1)

        if wacc_var.trend() > 0.01 and round > 10:
            print("Stalled")
            break

    print("End of iteration")


    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans
    import numpy as np

    for c in range(n_clusters):
        # perfect partitioning (using oracle information)

        client_partitions = [
            int(ds.dex.split("-")[1])
            for i, ds in enumerate(dss)
            if client_cluster[i] == c
        ]

        npartitions = np.unique(client_partitions)

        if len(npartitions) == 1:
            print(f"Cluster {c} has only one partition, skipping clustering")
            continue

        n_clusters += 1
        start_models.append(deep_clone_state_dict(start_models[c]))

        middle = len(npartitions) // 2
        new_clusters = npartitions[middle:]

        client_abs_index = [
            i
            for i, ds in enumerate(dss)
            if client_cluster[i] == c
        ]

        for cp in range(len(client_abs_index)):
            i = client_abs_index[cp]
            client_cluster[i] = n_clusters - 1
            print(f"Move client {i} from cluster {c} to {n_clusters - 1}")


     

    #     model_vecs = [
    #         wnd_dirs[i].avg()
    #         for i in range(len(dss))
    #         if client_cluster[i] == c
    #     ]

    #     model_vecs = [
    #         torch.cat([model_vecs[i][k].view(-1) for k in model_vecs[i]])
    #         for i in range(len(model_vecs))
    #     ]

    #     client_real_idx = [
    #         i
    #         for i in range(len(dss))
    #         if client_cluster[i] == c
    #     ]

    #     if len(model_vecs) <= 2:
    #         print(f"Cluster {c} has only {len(model_vecs)} models, skipping clustering")
    #         continue

    #     vecs_tensor = torch.stack(model_vecs)
    #     vecs = vecs_tensor.detach().cpu().numpy()
    #     vecs_normalized = normalize(vecs, norm="l2", axis=1)
    #     kmeans = KMeans(n_clusters=2, random_state=42)
    #     clusters = kmeans.fit_predict(vecs_normalized)

    #     start_models.append(deep_clone_state_dict(start_models[c]))
    #     n_clusters += 1


    #     indices = np.where(clusters == 1)[0]
    #     indices = [client_real_idx[i] for i in indices]
    #     for i in indices:
    #         print(f"Move client {i} from cluster {client_cluster[i]} to {n_clusters - 1}")
    #         client_cluster[i] = n_clusters - 1

    assert len(start_models) == n_clusters, f"Expected {n_clusters} models, got {len(start_models)}"
    assert len(client_cluster) == len(dss)

plt.pause(0.1)
plt.show(block=True)
