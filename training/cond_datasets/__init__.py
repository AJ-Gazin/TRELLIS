from typing import Union, List, Dict
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from .multiview_cond_dataset import (
    MultiviewConditonDataset,
    MixedMultiviewConditionDataset,
)
from .image_dataset import ImageDataset, MixedImageDataset
from .volume_cond_dataset import VolumeConditonDataset


def _create_single_multiview_dataset(db_config: Dict):
    return MultiviewConditonDataset(**db_config)


def _create_mixed_multiview_dataset(db_config: List):
    dbs = []
    db_weights = []
    for db_info in db_config:
        db_weight = db_info.pop("weight", 1.0)
        db = MultiviewConditonDataset(**db_info)

        dbs.append(db)
        db_weights.append(db_weight)
    return MixedMultiviewConditionDataset(dbs, db_weights)


def create_multiview_dataset(db_config: Union[DictConfig, ListConfig]):
    if isinstance(db_config, ListConfig):
        return _create_mixed_multiview_dataset(db_config)
    return _create_single_multiview_dataset(db_config)


def _create_image_dataset(db_config: Dict):
    return ImageDataset(**db_config)


def _create_mixed_image_dataset(db_config: List):
    dbs = []
    db_weights = []
    for db_info in db_config:
        db_weight = db_info.pop("weight", 1.0)
        db = ImageDataset(**db_info)

        dbs.append(db)
        db_weights.append(db_weight)
    return MixedImageDataset(dbs, db_weights)


def create_image_dataset(db_config: Union[DictConfig, ListConfig]):
    if isinstance(db_config, ListConfig):
        return _create_mixed_image_dataset(db_config)
    return _create_image_dataset(db_config)
