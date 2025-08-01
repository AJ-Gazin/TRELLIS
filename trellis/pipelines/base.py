from typing import *

import numpy as np
import torch
import torch.nn as nn

from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """

    def __init__(self, models: dict[str, nn.Module] = None, low_vram: bool = False):
        if models is None:
            return
        self.low_vram = low_vram
        self.models = models
        for model in self.models.values():
            model.eval()

    def unload_models(self, model_keys: Optional[List]):
        """
        Unload specified model(s) to save VRAM.
        """
        for model_key in model_keys:
            if model_key in self.models:
                if isinstance(self.models[model_key], dict):
                    for k, v in self.models[model_key].items():
                        if hasattr(v, "to"):
                            v.to(torch.device("cpu"))
                else:
                    if hasattr(self.models[model_key], "to"):
                        self.models[model_key].to(torch.device("cpu"))
        torch.cuda.empty_cache()

    def load_model(self, model_key: str):
        """
        Move a model to CUDA and return.
        """
        if isinstance(self.models[model_key], dict):
            for k, v in self.models[model_key].items():
                if hasattr(v, "to"):
                    v.to(torch.device("cuda"))
        else:
            if hasattr(self.models[model_key], "to"):
                self.models[model_key].to(torch.device("cuda"))
        return self.models[model_key]

    def verify_model_low_vram_devices(self):
        """
        Verify that all models are on the expected device.
        """
        keys_to_unload = [
            key
            for key in self.models.keys()
            if hasattr(self.models[key], "device")
            and self.models[key].device != torch.device("cpu")
        ]
        self.unload_models(keys_to_unload)

    @staticmethod
    def from_pretrained(path: str, cache_dir: str = "", skip_models: list = []) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import json
        import os

        if os.path.exists(f"{cache_dir}/pipeline.json"):
            path = cache_dir

        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download

            config_file = hf_hub_download(path, "pipeline.json", cache_dir=cache_dir)

        with open(config_file, "r") as f:
            args = json.load(f)["args"]

        _models = {}
        for k, v in args["models"].items():
            if k in skip_models:
                print('Model {} k is SKIPPED'.format(k))
                continue 
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except:
                _models[k] = models.from_pretrained(v)

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    def save_pretrained(self, save_dir: str):
        """
        Inverse logic against `from_pretrained` function
        Currently ignore the model key field
        """
        for k, model in self.models.items():
            print("saving the model {}...".format(k))
            if getattr(model, "save_pretrained", None):
                model.save_pretrained(f"{save_dir}/ckpts/")

    @property
    def device(self) -> torch.device:
        if self.low_vram:
            return "cuda"

        for model in self.models.values():
            if hasattr(model, "device"):
                return model.device
        for model in self.models.values():
            if hasattr(model, "parameters"):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            if isinstance(model, dict):
                for k, v in model.items():
                    if hasattr(v, "to"):
                        v.to(device)
            else:
                if hasattr(model, "to"):
                    model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))

    
    def preprocess_voxel(
        self, binary_voxel: np.ndarray, voxel_res: int = 64, concat_dim: bool = True
    ) -> torch.Tensor:
        """
        Preprocess(read / voxelize) the given 3D object.
        """
        assert all(
            [s == voxel_res for s in binary_voxel.shape]
        ), "Input voxels have incompatible resolution {}".format(binary_voxel.shape)
        # Active voxels (N_p x 3)
        x, y, z = np.nonzero(binary_voxel)
        values_sum = x * voxel_res * voxel_res + y * voxel_res + z
        active_voxels = np.stack([x, y, z], axis=1)[np.argsort(values_sum)]
        if not concat_dim:
            return torch.from_numpy(active_voxels).int().cuda()
        
        active_voxels = np.concatenate(
            (np.zeros((len(active_voxels), 1), dtype=np.int32), active_voxels), axis=1
        )
        # Pad with the batch dimension
        return torch.from_numpy(active_voxels).int().cuda()
