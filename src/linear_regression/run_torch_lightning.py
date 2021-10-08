from sklearn.model_selection import train_test_split 
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn 
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as plt 
import pandas as pd 
import torch
from typing import Tuple, Optional


class LinearRegression(nn.Module):
    """
    """
    def __init__(self, input_dim: int=13, output_dim: int=1):
        """
        """
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        return self.model(x)


class HousingDataModule(plt.LightningDataModule):
    """
    """
    def __init__(self,
            data_dir: str='../../data/housing.csv',
            dataset_split: Tuple[float, float]=(55_000, 5_000, 10_000),
            batch_size: int=32,
            num_workers: int=0,
            pin_memory: bool=False
        ):
        super().__init__()
        self.save_hyperparameters()

        # MNIST Dataset object 
        self.data_train: Optional[Dataset] = None 
        self.data_val: Optional[Dataset] = None 


    def prepare_data(self) -> None:
        """
        """
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        data = pd.read_csv(self.hparams.data_dir, delimiter=r"\s+", names=column_names)
        Y = data['MEDV']
        X = data.drop('MEDV', axis=1)
        # X = pd.DataFrame(np.c_[data['LSTAT'], data['RM']], columns = ['LSTAT','RM'])
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)
        X_train, Y_train = torch.tensor(X_train.values, dtype=torch.float), torch.tensor(Y_train.values, dtype=torch.float).reshape(-1, 1)
        X_val, Y_val = torch.tensor(X_val.values, dtype=torch.float), torch.tensor(Y_val.values, dtype=torch.float).reshape(-1, 1)
        self.data_train, self.data_val = TensorDataset(X_train, Y_train), TensorDataset(X_val, Y_val)
        self.input_dim = X_train.shape[1]
        self.output_dim = Y_train.shape[1]


    def train_dataloader(self):
        """
        """
        return DataLoader(
            dataset = self.data_train,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = True
        )

    def val_dataloader(self):
        """
        """
        return DataLoader(
            dataset = self.data_val,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers,
            pin_memory = self.hparams.pin_memory, 
            shuffle = False
        )

    def test_dataloader(self):
        """
        """
        raise NotImplemented



class LitLinearRegression(plt.LightningModule):
    """
    """
    def __init__(self, input_dim: int=13, output_dim: int=1, learning_rate: float=0.01):
        """
        """
        super().__init__()
        self.save_hyperparameters()
        self.backbone = LinearRegression(input_dim=input_dim, output_dim=output_dim)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        return self.backbone(x)

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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss 

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        """
        x, y  = batch 
        loss = self.criterion(self(x), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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
    epochs = 200 
    learning_rate = 0.01 

    # Dataset 
    housing_data = HousingDataModule(data_dir='../../data/housing.csv', batch_size=batch_size)
    housing_data.prepare_data()

    # Model 
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="model",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=5,
        mode="min",
    )
    model = LitLinearRegression(input_dim=housing_data.input_dim, output_dim=housing_data.output_dim, learning_rate=learning_rate)

    # Training 
    trainer = plt.Trainer(
        gpus=None,
        max_epochs=epochs,
        progress_bar_refresh_rate=20,
        callbacks=[checkpoint_callback]
    )
    # result = trainer.fit(model, train_loader, val_loader)
    result = trainer.fit(model, housing_data)

    print(result)

if __name__ == '__main__':
    """
    """
    main()
