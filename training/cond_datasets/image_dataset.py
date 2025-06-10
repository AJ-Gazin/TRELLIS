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
from trellis.modules.sparse import SparseTensor


class ImageDataset(Dataset):
    def __init__(
        self,
        name: str,
        inroot: str,
        num_views: int = 4,
        num_max_instances: Optional[int] = None,
        get_eval: bool = False,
        get_image: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()
        # only request the fields we DO need
        self._get_eval = get_eval
        self._get_image = get_image
        self._num_views = num_views
        self._request_fields = ["ss_latent", "slat_latent"]

        if self._get_eval:
            self._request_fields.append("eval_img")
        else:
            self._request_fields.append("cond_img")

        self._provider = UnifiedProvider(
            name, inroot, request_fields=self._request_fields
        )
        self._instances_with_ids = self._provider.get_valid_instance_with_ids(
            extra_key="eval_img" if self._get_eval else "cond_img",
            num_views=self._num_views,
            num_max_instances=num_max_instances,
            verbose=True,
        )
        self._verbose = verbose
        self._image_transform = transforms.Compose([transforms.Resize((518, 518))])

    def __getitem__(self, index):
        instance_name, instance_view = self._instances_with_ids[index]
        instance_caption = self._provider.get_caption(instance_name)
        if self._verbose and not instance_caption:
            warnings.warn(f"Instance {instance_caption} does not have caption!!!")

        if self._get_eval:
            instance_dir = osp.join(
                self._provider.get_field_dir("eval_img"), instance_name
            )
            caminfo = self._provider.get_eval_img_transforms(instance_name)
        else:
            instance_dir = osp.join(
                self._provider.get_field_dir("cond_img"), instance_name
            )
            caminfo = self._provider.get_cond_img_transforms(instance_name)
        # Take the first frame
        frame_info = parse_trellis_frame(
            instance_dir, caminfo["frames"][instance_view], get_image=self._get_image
        )

        extrinsics = frame_info["extrinsics"].float()
        intrinsics = frame_info["intrinsics"].float()

        # Get ss latent
        ss_latent = torch.from_numpy(
            self._provider.get_ss_latent(instance_name)
        ).float()
        slat_coords, slat_feats = self._provider.get_slat_latent(instance_name)
        slat_coords, slat_feats = (
            torch.from_numpy(slat_coords).int(),
            torch.from_numpy(slat_feats).float(),
        )
        # Concatenate one column, which will be updated in the customized collate_fn in dataloader
        slat_coords = torch.cat(
            (torch.zeros((len(slat_coords), 1)).type_as(slat_coords), slat_coords),
            dim=1,
        )
        slat_latent = SparseTensor(
            feats=slat_feats,
            coords=slat_coords,
        )

        data = {
            "name": instance_name,
            "caption": instance_caption,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "ss_latent": ss_latent,
            "slat_latent": slat_latent,
        }

        if self._get_image:
            data["image"] = self._image_transform(frame_info["image"].float())
        return data

    def __len__(self):
        return len(self._instances_with_ids)


class MixedImageDataset(Dataset):
    def __init__(self, dbs: List[ImageDataset], weights: Optional[List[float]] = None):
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

        # Randomly select a dataset based on weights
        dataset_idx = np.random.choice(
            len(self._dbs), p=np.array(self._weights) / sum(self._weights)
        )
        # Randomly select a sample from the chosen dataset
        sample_idx = np.random.randint(0, len(self._dbs[dataset_idx]))

        return self._dbs[dataset_idx][sample_idx]

    def __len__(self):
        return self._total_size
