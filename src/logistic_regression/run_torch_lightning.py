from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from torch import nn 
from typing import Optional, List
import pytorch_lightning as plt
import pandas as pd
import torch


class CIFAR10Dataset(Dataset):
    """
    """
    def __init__(self,
            classes: List[str]=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
            'frog', 'horse', 'ship', 'truck'],
            data_path: str='../../data/cifar-10', 
            stage: str='train',
            transform: transforms=None
        ):
        super().__init__()
        self.classes = {v: i for i, v in enumerate(classes)}
        self.data_path = Path(data_path)
        self.stage = stage 
        self.transform = transform
        self._read_csv()

    def _read_csv(self):
        """
        """
        if self.stage == 'train':
            data = pd.read_csv(self.data_path / 'trainLabels.csv')
        elif self.stage == 'test':
            data = pd.read_csv(self.data_path / 'test.csv')
        self.image_names = list(data['id'])
        self.labels = [self.classes[i] for i in data['label']]

    def __getitem__(self, index: int):
        filename = self.image_names[index]
        label = self.labels[index]
        image_path = self.data_path / self.stage / '{}.png'.format(filename)
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img.view(-1), label

    def __len__(self):
        return len(self.labels)

class CIFAR10LitDataModule(plt.LightningDataModule):
    """
    """
    def __init__(self,
            data_path: str='../../data/cifar-10',
            batch_size: int=32,
            input_shape: tuple=(32, 32, 3),
            mean: list=[0.4914, 0.4822, 0.4465],
            std: list=[0.2470, 0.2435, 0.2616],
            num_workers: int=0,
            pin_memory: bool=False
        ):
        super().__init__()
        self.save_hyperparameters()
        self.train_transforms = transforms.Compose([ 
            transforms.RandomResizedCrop(input_shape[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.val_transforms = transforms.Compose([ 
            transforms.Resize(input_shape[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.data_train: Optional[Dataset] = None 
        self.data_val: Optional[Dataset] = None 

    def setup(self, stage: Optional[str]):
        """
        """
        if stage in ('fit', None):
            self.data_train = CIFAR10Dataset(
                data_path=self.hparams.data_path,
                stage='train',
                transform=self.train_transforms
            )
            self.data_val = CIFAR10Dataset(
                data_path=self.hparams.data_path,
                stage='test',
                transform=self.val_transforms
            )

        if stage in ('test', None):
            pass 

    def train_dataloader(self):
        return DataLoader(
            dataset = self.data_train,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = True
        ) 

    def val_dataloader(self):
        return DataLoader(
            dataset = self.data_val,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers,
            pin_memory = self.hparams.pin_memory, 
            shuffle = False
        )

class LogisticRegression(nn.Module):
    """
    """
    def __init__(self, input_dim: int=32*32*3, output_dim: int=10):
        """
        """
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)

class LitLogisticRegression(plt.LightningModule):
    """
    """
    def __init__(self, input_dim: int=32*32*3, output_dim: int=10, learning_rate: float=0.01):
        """
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = LogisticRegression(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        """
        return self.model(x)

    def on_epoch_start(self):
        """
        """
        print("\n")

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        """
        x, y = batch 
        logit = self(x)
        loss = self.criterion(logit, y)
        _, pred = torch.max(logit, 1)
        acc = (pred==y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        """
        x, y = batch 
        logit = self(x)
        loss = self.criterion(logit, y)
        _, pred = torch.max(logit, 1)
        acc = (pred==y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """
        """
        raise NotImplemented

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        

def main():
    """
    """
    # Hyperparameters 
    batch_size = 32
    epochs = 5 
    learning_rate = 0.01 
    result_dir = 'model'
    h, w, c = 32, 32, 3
    num_classes = 10

    # Dataset 
    cifar10_data = CIFAR10LitDataModule(
        data_path='../../data/cifar-10',
        input_shape=(h, w, c), 
        batch_size=batch_size
    )

    # Model 
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=result_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=5,
        mode="min",
    )
    model = LitLogisticRegression(input_dim=h*w*c, output_dim=num_classes, learning_rate=learning_rate)

    # Training 
    trainer = plt.Trainer(
        gpus=None, 
        max_epochs=epochs,
        progress_bar_refresh_rate=20,
        callbacks=[checkpoint_callback]
    )
    result = trainer.fit(model, cifar10_data)

if __name__ == '__main__':
    main()