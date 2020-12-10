from typing import Optional

import numpy as np
from torch.utils.data import random_split, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_lightning import LightningDataModule

from PrecipitationForecastRadar.dataset import dataset_jma_radar

# https://github.com/PyTorchLightning/pytorch-lightning/blob/94a9d3d2837eb962cb47ad2854569039a552f729/pl_examples/basic_examples/mnist_datamodule.py
# https://nikkie-ftnext.hatenablog.com/#Data-PyTorch-Dataloader--LightningDataModule
# https://optie.hatenablog.com/

dataset_classes = {
    'PrecipitationJMADataset': dataset_jma_radar.PrecipitationJMADataset
}

class PrecipRegressionDataModule(LightningDataModule):
    def __init__(self, dataset_name='PrecipitationJMADataset', batch_size=32, num_workers=0, shuffle=False,
                 valid_size: float = 0.1, **kwargs):
        super().__init__()
        self.Dataset = dataset_classes[dataset_name]
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.valid_size = valid_size

        self.arg_train = 'train'
        # todo: add args of train_val_split for change val -> random
    # for downloading
    def prepare_data(self):
        self.Dataset(**self.kwargs)

    def setup(self, stage: Optional[str] = None):

        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None:
            self.kwargs[self.arg_train] = True
            self.dataset_train = self.Dataset(**self.kwargs)
            self.dataset_val = self.Dataset(**self.kwargs)
            self.dims = self.dataset_train[0][0].shape

            num_train = len(self.dataset_train)
            indices = list(range(num_train))
            split = int(np.floor(self.valid_size * num_train))

            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.valid_sampler = SubsetRandomSampler(valid_idx)

        if stage == 'test' or stage is None:
            self.kwargs[self.arg_train] = False
            self.dataset_test = self.Dataset(**self.kwargs)
            self.dims = self.dataset_test[0][0].shape

    def train_dataloader(self):
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            sampler=self.valid_sampler,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

