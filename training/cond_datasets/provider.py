import os, os.path as osp
import glob
import json
import numpy as np
import utils3d
import pandas as pd
from functools import reduce
from typing import Union, List, Optional, Literal
from torch.utils.data import Dataset

SS_ENCODER_NAME = "ss_enc_conv3d_16l8_fp16"
SLAT_ENCODER_NAME = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"


class UnifiedProvider:

    def __init__(
        self,
        name: str,
        inroot: str,
        request_fields: Optional[List] = None,
        ss_encoder_name: str = SS_ENCODER_NAME,
        slat_encoder_name: str = SLAT_ENCODER_NAME,
    ):
        super().__init__()
        # dataset name
        self._name = name
        self._metadata_path = osp.join(inroot, "metadata.csv")
        assert osp.exists(
            self._metadata_path
        ), f"Can not find meta data path: {self._metadata_path}"
        self._metadata = pd.read_csv(self._metadata_path)
        self._name2caption = {
            name: caption
            for name, caption in zip(
                list(self._metadata["sha256"].values),
                list(self._metadata["captions"].values),
            )
        }
        # preprocessed items
        self._feature_dir = osp.join(inroot, "features")
        self._ss_latent_dir = osp.join(inroot, "ss_latents", ss_encoder_name)
        self._slat_latent_dir = osp.join(inroot, "latents", slat_encoder_name)
        self._voxel_dir = osp.join(inroot, "voxels")
        self._img_dir = osp.join(inroot, "renders")
        self._cond_img_dir = osp.join(inroot, "renders_cond")
        self._eval_img_dir = osp.join(inroot, "renders_eval")
        self._request_fields = request_fields
        for field in self._request_fields:
            self._verify_field(field)

    def get_field_dir(
        self,
        field_name: Literal[
            "feature",
            "ss_latent",
            "slat_latent",
            "voxel",
            "img",
            "cond_img",
            "eval_img",
        ],
    ):
        return getattr(self, f"_{field_name}_dir")

    def _verify_field(self, field_name: str):
        assert osp.exists(
            self.get_field_dir(field_name)
        ), f"Directory {self.get_field_dir(field_name)} for item {field_name} does not exist!"

    def get_caption(self, sha256: str):
        return self._name2caption.get(sha256, "")

    def get_valid_instance_with_ids(
        self,
        extra_key: str = "cond_img",
        num_views: int = 4,
        num_max_instances: Optional[int] = None,
        verbose: bool = False,
    ):
        valid_instance_names = self.get_valid_instances(
            extra_key=extra_key, verbose=verbose, num_max_instances=num_max_instances
        )
        valid_instance_with_ids = []
        for valid_instance in valid_instance_names:
            instance_dir = getattr(self, f"_{extra_key}_dir")
            instance_images = glob.glob(osp.join(instance_dir, valid_instance, "*.png"))
            num_instance_views = len(instance_images)
            # instance having less than num views is invalid
            if num_instance_views < num_views:
                continue
            for i in range(min(num_views, len(instance_images))):
                valid_instance_with_ids.append((valid_instance, i))

        if verbose:
            print(
                "[{}] having {} valid images finally (under {} views constraint).".format(
                    self._name, len(valid_instance_with_ids), num_views
                )
            )
        return valid_instance_with_ids

    def get_valid_instances(
        self,
        extra_key: str = "cond_img",
        verbose: bool = False,
        num_max_instances: Optional[int] = None,
    ):
        get_instance_name = lambda file_name: osp.splitext(file_name)[0]
        instances = {}

        for field in self._request_fields:
            field_dir = self.get_field_dir(field)
            field_instances = set(map(get_instance_name, os.listdir(field_dir)))
            if field in ["img", extra_key]:
                field_instances = set(
                    filter(
                        lambda instance: osp.exists(
                            osp.join(field_dir, instance, "transforms.json")
                        ),
                        field_instances,
                    )
                )
            instances[field] = field_instances

        # reduction to common ones
        valid_instance_names = list(reduce(set.intersection, list(instances.values())))
        if verbose:
            for field, field_instances in instances.items():
                print(
                    "[{}] Field {} has {} instances.".format(
                        self._name, field, len(field_instances)
                    )
                )
            print(
                "[{}] having {} valid instances finally.".format(
                    self._name, len(valid_instance_names)
                )
            )
        if num_max_instances is not None:
            if verbose:
                print(
                    "[{}] Truncation to {} instances.".format(
                        self._name, num_max_instances
                    )
                )
            valid_instance_names = valid_instance_names[:num_max_instances]
        return valid_instance_names

    def get_img_path(self, instance: str, index: int):
        return osp.join(self._img_dir, instance, f"{index:03d}.png")

    def get_img_transforms(self, instance: str):
        return json.load(osp.join(self._img_dir, instance, "transforms.json"))

    def get_cond_img_path(self, instance, index):
        return osp.join(self._cond_img_dir, instance, f"{index:03d}.png")

    def get_cond_img_transforms(self, instance: str):
        return json.load(
            open(osp.join(self._cond_img_dir, instance, "transforms.json"), "r")
        )

    def get_eval_img_path(self, instance, index):
        return osp.join(self._eval_img_dir, instance, f"{index:03d}.png")

    def get_eval_img_transforms(self, instance: str):
        return json.load(
            open(osp.join(self._eval_img_dir, instance, "transforms.json"), "r")
        )

    def get_ss_latent(self, instance: str):
        npz_path = osp.join(self._ss_latent_dir, instance + ".npz")
        ss_latent = np.load(npz_path)
        return ss_latent["mean"]

    def get_slat_latent(self, instance: str):
        npz_path = osp.join(self._slat_latent_dir, instance + ".npz")
        slat_latent = np.load(npz_path)
        return slat_latent["coords"], slat_latent["feats"]

    def get_voxel(self, instance: str):
        voxel_path = osp.join(self._voxel_dir, instance + ".ply")
        return utils3d.io.read_ply(voxel_path)[0]

    @property
    def name(self):
        return self._name

    @property
    def request_fields(self):
        return self._request_fields
