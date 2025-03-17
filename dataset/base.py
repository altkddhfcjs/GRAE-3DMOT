import torch
import itertools
from typing import List, Optional, Sequence, Union
from models.structures import Instances


class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        elem = batch[0]
        return [s for s in zip(*batch)]
        # if isinstance(elem, Instances):
        #     return batch
        # elif isinstance(elem, float):
        #     return torch.tensor(batch, dtype=torch.float)
        # elif isinstance(elem, int):
        #     return torch.tensor(batch)
        # elif isinstance(elem, str):
        #     return batch
        # elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        #     return type(elem)(*(self(s) for s in zip(*batch)))
        # elif isinstance(elem, List) and hasattr(elem, '_fields'):
        #     return [s for s in zip(*batch)]
        # raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # pragma: no cover
        # TODO Deprecated, remove soon.
        return self(batch)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch=None,
        exclude_keys=None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )