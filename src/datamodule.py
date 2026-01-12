import lightning as L

from torch.utils.data import DataLoader

from .data_utils import CachedImageFolder


class CachedLatentsDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None

    def setup(self, stage=None):
        self.train_dataset = CachedImageFolder(self.data_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
