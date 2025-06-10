import os, os.path as osp
import glob
import warnings
import numpy as np
import torch
from PIL import Image
from typing import Union, List, Optional
from torch.utils.data import Dataset
from torchvision import transforms
from .provider import UnifiedProvider
from .data_utils import parse_trellis_frame


class MultiviewConditonDataset(Dataset):

    def __init__(
        self,
        name: str,
        inroot: str,
        min_views: int = 2,
        max_views: int = 4,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()
        # only request the fields we DO need
        self._provider = UnifiedProvider(
            name, inroot, request_fields=["cond_img", "ss_latent", "slat_latent"]
        )
        self._instance_names = self._provider.get_valid_instances(verbose=True)
        self._min_views = min_views
        self._max_views = max_views
        self._verbose = verbose
        self._image_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                # Same resolution as trellis training
                transforms.Resize(518, 518),
            ]
        )

    def __getitem__(self, index):
        instance_name = self._instance_names[index]
        instance_caption = self._provider.get_caption(instance_name)
        instance_dir = osp.join(self._provider.get_field_dir("cond_img"), instance_name)
        if self.verbose and not instance_caption:
            warnings.warn(f"Instance {instance_caption} does not have caption!!!")

        caminfo = self._provider.get_cond_img_transforms(instance_name)
        # randomly sample indices
        num_views = np.random.randint(self._min_views, self._max_views + 1)
        frame_indices = np.random.randint(0, len(caminfo["frames"]), num_views)
        frame_info = [
            parse_trellis_frame(instance_dir, caminfo["frames"][frame_index])
            for frame_index in frame_indices
        ]
        images = self._image_transform(
            torch.stack([frame["image"] for frame in frame_info], axis=0).float().cuda()
        )
        # Normalise to 0 ~ 1 is OK
        images = images / 255.0

        extrinsics = (
            torch.from_numpy(
                np.stack([frame["extrinsics"] for frame in frame_info], axis=0)
            )
            .float()
            .cuda()
        )
        intrinsics = (
            torch.from_numpy(
                np.stack([frame["intrinsics"] for frame in frame_info], axis=0)
            )
            .float()
            .cuda()
        )
        # pad all
        _, c, h, w = images.shape
        # TODO: improve this
        images = torch.cat(
            [images, torch.zeros(self._max_views - num_views, c, h, w).type_as(images)],
            dim=0,
        )
        extrinsics = torch.cat(
            [
                extrinsics,
                torch.zeros(self._max_views - num_views, 4, 4).type_as(extrinsics),
            ],
            dim=0,
        )
        intrinsics = torch.cat(
            [
                intrinsics,
                torch.zeros(self._max_views - num_views, 3, 3).type_as(intrinsics),
            ],
            dim=0,
        )

        # slat and ss
        ss_latent = (
            torch.from_numpy(self._provider.get_ss_latent(instance_name)).float().cuda()
        )
        # slat_latent = self._provider.get_slat_latent(instance)
        return {
            "name": instance_name,
            "caption": instance_caption,
            "num_views": num_views,
            "images": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "ss_latent": ss_latent,
            # "slat_latent": slat_latent
        }

    def __len__(self):
        return len(self._instance_names)


class MixedMultiviewConditionDataset(Dataset):

    def __init__(
        self, dbs: List[MultiviewConditonDataset], weights: Optional[List[float]] = None
    ):
        super().__init__()
        self._dbs = dbs
        self._weights = weights if weights is not None else [1.0] * len(dbs)
        self._cumulative_sizes = self._compute_cumulative_sizes()
        self._total_size = self._cumulative_sizes[-1]

    def _compute_cumulative_sizes(self):
        sizes = [len(dataset) for dataset in self._dbs]
        return np.cumsum([0] + sizes)

    def __getitem__(self, index):
        if index < 0 or index >= self._total_size:
            raise IndexError("Index out of range")

        dataset_idx = np.random.choice(
            len(self._dbs), p=np.array(self._weights) / sum(self._weights)
        )
        sample_idx = np.random.randint(0, len(self._dbs[dataset_idx]))

        return self._dbs[dataset_idx][sample_idx]

    def __len__(self):
        return self._total_size
