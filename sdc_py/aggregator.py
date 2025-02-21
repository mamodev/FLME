import torch
import protocol
from river import cluster
from river import stream

import numpy as np

from sklearn.decomposition import IncrementalPCA 
from sklearn.cluster import MiniBatchKMeans

class WindMetric:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = []
        self.prev_avg = None

    def add(self, value):
        self.prev_avg = self.get_avg()

        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def get_avg(self):
        if len(self.window) == 0:
            return +float('inf')
        
        return sum(self.window) / len(self.window)

    # returns a number in percentage (0-100)
    def get_avg_change(self):
        if self.prev_avg is None:
            return 1

        return 100 * (self.prev_avg - self.get_avg()) / self.prev_avg
    
    def get_variance(self):
        if len(self.window) == 0:
            return +float('inf')
        
        avg = self.get_avg()
        return sum((x - avg) ** 2 for x in self.window) / len(self.window)
    

    def __str__(self):
        return f"avg: {self.get_avg():.4f}, avg_change: {self.get_avg_change():.2f}%, variance: {self.get_variance():.4f}"

class Aggregator:
    def __init__(self, initial_model, on_new_model, cluster_id):

        self.M_train_loss = WindMetric(10)
        self.M_test_loss = WindMetric(10)
        self.M_model_delta = WindMetric(10)

        self.cluster_id = cluster_id
        self.current_model_version = 1
        self.updates = []
        self.on_new_model = on_new_model
        self.FED_BUFF_THRESHOLD = 20
        self.initial_model = initial_model
        self.gmodel = initial_model

        self.splitting = False
        # self.dbstream: cluster.DBSTREAM = None
        # self.reducer = KernelPCA(n_components=10, kernel="rbf", eigen_solver="auto")
        # self.reducer = PCA(n_components=10)
        # self.reducer = IncrementalPCA(n_components=10, batch_size=10)
        self.reducer : IncrementalPCA = None
        self.kmeans : MiniBatchKMeans = None
        self.splitting_debounce = 0

    def split_learn(self):
        flats = [self.flat_model(model).numpy() for _, _, _, model in self.updates]
        flats = np.vstack(flats)

        if self.kmeans is None:
            self.kmeans = MiniBatchKMeans(n_clusters=2, random_state=0)
            self.reducer = IncrementalPCA(n_components=10, batch_size=10)
            self.reducer.partial_fit(flats)
            print(f"PCA FITTED")


        # flats = self.reducer.transform(flats)
        self.kmeans.partial_fit(flats)
        print(f"KMEANS: {self.kmeans.cluster_centers_}")

        # self.dbstream = cluster.DBSTREAM(
            #     cleanup_interval=100,
            #     fading_factor=0.1,
            #     clustering_threshold=0.1,
            #     intersection_factor=0.5,
            #     minimum_weight=1,
            # )

        # print("Learning from updates")
        # for flat in flats:
        #     print("Learning from update")
        #     flat_dict = {i: flat[i] for i in range(len(flat))}  # Convert array to dict
        #     print("Transfomed to dict")
        #     self.dbstream.learn_one(flat_dict)

        # print(f"DBSTREAM: {self.dbstream.centers}")

    def put(self, wid, client_key, meta, state_buffer):
        model, _ = protocol.ModelData.from_buffer(state_buffer)
        model = model.model_state

        self.updates.append((wid, client_key, meta, model))
        assert len(self.updates) <= self.FED_BUFF_THRESHOLD, "Too many updates in buffer" 
        
        split = None
      
        if len(self.updates) == self.FED_BUFF_THRESHOLD:
            self.filter_updates()
            if len(self.updates) != self.FED_BUFF_THRESHOLD:
                return None


            self.current_model_version += 1
            print(f"Updating global model to version {self.current_model_version}")

            self.filter_updates()
            mean_train_loss, mean_test_loss = self.compute_loss_means()

            self.M_model_delta.add(self.compute_updates_delta())
            self.M_train_loss.add(mean_train_loss)
            self.M_test_loss.add(mean_test_loss)

            print(f"Mean train loss: {mean_train_loss:.4f}, Mean test loss: {mean_test_loss:.4f}")
            print(f"Train loss: {self.M_train_loss}")
            print(f"Test loss: {self.M_test_loss}")
            print(f"Model delta: {self.M_model_delta}")
        
           


            if self.splitting_debounce > 0:
                self.splitting_debounce -= 1

            if self.splitting_debounce == 0:
                should_start_split_learn, should_split = self.should_split()
                if should_start_split_learn and not self.splitting:
                    self.splitting = True

                if self.splitting:
                    self.split_learn()

                if should_split:
                    split = [self.unflat_model(c) for c in self.kmeans.cluster_centers_]
                    self.splitting = False
                    self.kmeans = None
                    self.reducer = None
                    self.splitting_debounce = 10

                    self.gmodel = split[0]
                    self.updates.clear()

                    return split
            

            new_model = self.fed_avg()
            self.on_new_model(self.cluster_id, self.current_model_version, new_model)
            self.gmodel = new_model
            self.updates.clear()

        return split


    # returns (bool, bool) - (should_start_split_learn, split!)
    def should_split(self):
        tloss = self.M_train_loss.get_avg()
        tchang = self.M_train_loss.get_avg_change()
        eloss = self.M_test_loss.get_avg()
        echang = self.M_test_loss.get_avg_change()

        d = self.M_model_delta.get_avg()
        dchange = self.M_model_delta.get_avg_change()


        if (tchang < 1 or tloss < 0.01) and eloss > 2 * tloss and (dchange < 1 or d < 0.1):
            return (False, True)
        
        if (tchang < 5 or tloss < 0.1) and eloss > 2 * tloss:
            return (True, False)
        
        return (False, False)

    def flat_model(self, model):
        return torch.cat([v.flatten() for v in model.values()])
    
    def unflat_model(self, flat_model):
        model = {}
        idx = 0
        for k, v in self.initial_model.items():
            model[k] = flat_model[idx:idx + v.numel()].reshape(v.shape)
            idx += v.numel()
        
        assert idx == len(flat_model), "Invalid model size"

        model = {k: torch.tensor(v) for k, v in model.items()}

        return model
    
    def compute_updates_delta(self):
        gmodel_flat = self.flat_model(self.gmodel)
        updates_flat = [self.flat_model(model) for _, _, _, model in self.updates]
        diff = [torch.norm(gmodel_flat - u) for u in updates_flat]
        return sum(diff) / len(diff)

    # in place modification
    def filter_updates(self):
        self.updates.reverse()
        unique_updates = []
        seen_keys = set()
        for wid, client_key, meta, model in self.updates:
            if client_key not in seen_keys:
                unique_updates.append((wid, client_key, meta, model))
                seen_keys.add(client_key)

        self.updates.clear()
        self.updates = unique_updates

    def compute_loss_means(self):
        tot_train_loss = 0
        tot_test_loss = 0
        for wid, client_key, meta, model in self.updates:
            tot_train_loss += meta.train_loss
            tot_test_loss += meta.test_loss

        mean_train_loss = tot_train_loss / len(self.updates)
        mean_test_loss = tot_test_loss / len(self.updates)

        return mean_train_loss, mean_test_loss

    def zero_model(self):
        return {k: torch.zeros_like(v) for k, v in self.initial_model.items()}

    def fed_avg(self):
        new_model = self.zero_model()
        tot_w = 0
        for wid, client_key, meta, model in self.updates:
            for k, v in model.items():
                new_model[k] += v * meta.train_samples
            tot_w += meta.train_samples

        new_model = {k: v / tot_w for k, v in new_model.items()}
        return new_model

