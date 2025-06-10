import torch
from cond_datasets import MultiviewConditonDataset

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    db_name = "ABO"
    db_root = "/root/TRELLIS/datasets/ABO"
    dataset = MultiviewConditonDataset(db_name, db_root)
    print(len(dataset))

    loader = DataLoader(dataset, batch_size=1)

    data = next(iter(loader))

    for k, v in data.items():
        print(k, v.shape if isinstance(v, torch.Tensor) else v)
