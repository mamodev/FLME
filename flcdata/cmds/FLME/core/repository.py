# import os
# from torch import save, load

# class ModelRepository:
#     def __init__(self, models: list, start_version: int, window_size: int, folder="./repo"):
#         self.folder = folder
#         self.models = models
#         self.start_version = start_version
#         self.window_size = window_size

#     def latest_model_version(self):
#         return self.start_version + len(self.models) - 1
    
#     def get_model(self, version: int):
#         assert self.start_version > 0, f"Repo not correctly initialized"

#         if version == 0:
#             version = self.latest_model_version()

#         if version > self.latest_model_version():
#             return None, None, "Model version is ahead of the latest model"
        
#         if version < self.start_version:
#             return None, None, "Model version is too old, model has been deleted"

#         return self.models[version - self.start_version], version, None
    
#     def put_model(self, model):
#         if len(self.models) == self.window_size:
#             self.models.pop(0)
#             self.start_version += 1

#         if len(self.models) == 0:
#             self.start_version = 1

#         self.models.append(model)   
#         ModelRepository.save_model(f"{self.folder}/{self.latest_model_version()}.model", model)

#     def save_model(path: str, model):
#         save(model, path)

#     def load_model(path: str):
#         return load(path)

#     def from_disk(folder: str, ignore_exists = True, window_size=10) -> "ModelRepository":
#         # check if folder exists
#         if not os.path.exists(folder):
#             if not ignore_exists:
#                 raise Exception(f"Folder {folder} does not exist")
#             os.makedirs(folder)

#         # check if folder is a directory
#         if not os.path.isdir(folder):
#             raise Exception(f"Path {folder} is not a directory")
        
#         versions = []
#         for file in os.listdir(folder):
#             if file.endswith(".model"):
#                 version = int(file.split(".")[0])
#                 versions.append(version)

#         end_idx = len(versions) 
#         start_idx = max(0, end_idx - window_size)
#         if len(versions) > 0:
#             versions = sorted(versions)[start_idx:end_idx]
#             start_version = versions[0]
#         else:
#             versions = []
#             start_version = 0

#         models = []
#         for version in versions:
#             path = f"{folder}/{version}.model"
#             model = ModelRepository.load_model(path)
#             models.append(model)

#         return ModelRepository(models, start_version, window_size, folder)


import os
from torch import save, load

class ModelRepository:
    def __init__(
        self,
        models: list[tuple],
        start_version: int,
        window_size: int,
        folder="./repo",
    ):
        self.folder = folder
        self.models = models
        self.start_version = start_version
        self.window_size = window_size

    def latest_model_version(self):
        return self.start_version + len(self.models) - 1

    def get_model(self, version: int):
        assert self.start_version > 0, "Repo not correctly initialized"

        if version == 0:
            version = self.latest_model_version()

        if version > self.latest_model_version():
            return (
                None,
                None,
                None,
                "Model version is ahead of the latest model",
            )

        if version < self.start_version:
            return (
                None,
                None,
                None,
                "Model version is too old, model has been deleted",
            )

        model, _ = self.models[version - self.start_version]
        return model, version, None

    def put_model(self, model, metrics=None):
        if metrics is None:
            metrics = {}

        if len(self.models) == self.window_size:
            self.models.pop(0)
            self.start_version += 1

        if len(self.models) == 0:
            self.start_version = 1

        self.models.append((model, metrics))
        ModelRepository.save_model(
            f"{self.folder}/{self.latest_model_version()}.model", model, metrics
        )

    def save_model(path: str, model, metrics: dict):
        save({"model": model, "metrics": metrics}, path)

    def load_model(path: str):
        checkpoint = load(path)
        return checkpoint["model"], checkpoint["metrics"]

    def from_disk(
        folder: str, ignore_exists=True, window_size=10
    ) -> "ModelRepository":
        # check if folder exists
        if os.path.exists(folder):
            if not ignore_exists:
                raise Exception(f"Folder {folder} exist")
        else: 
            os.makedirs(folder)

        # check if folder is a directory
        if not os.path.isdir(folder):
            raise Exception(f"Path {folder} is not a directory")

        versions = []
        for file in os.listdir(folder):
            if file.endswith(".model"):
                version = int(file.split(".")[0])
                versions.append(version)

        end_idx = len(versions)
        start_idx = max(0, end_idx - window_size)
        if len(versions) > 0:
            versions = sorted(versions)[start_idx:end_idx]
            start_version = versions[0]
        else:
            versions = []
            start_version = 0

        models = []
        for version in versions:
            path = f"{folder}/{version}.model"
            model, metrics = ModelRepository.load_model(path)
            models.append((model, metrics))

        return ModelRepository(models, start_version, window_size, folder)
