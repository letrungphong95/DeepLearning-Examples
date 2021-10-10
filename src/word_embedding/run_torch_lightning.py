from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as plt 
import collections
import torch


class WikiDataset(Dataset):
    """
    """ 
    def __init__(self, 
            data_path: str='../../data/text8',
            window_size: int=2,
            max_size: int=500000,
            min_occurence: int=10
        ):
        super().__init__()
        self.word_pair, self.vocab, self.word_to_idx, self.vocab_size = self._read_corpus(
            data_path=data_path,
            max_size=max_size,
            min_occurrence=min_occurence,
            window_size=window_size
        )

    def __getitem__(self, index: int):
        """
        """
        x = torch.tensor(self.word_to_idx[self.word_pair[index][0]])
        y = torch.tensor(self.word_to_idx[self.word_pair[index][1]])
        return x, y

    def __len__(self):
        """
        """
        return len(self.word_pair)

    def _read_corpus(self, data_path, max_size, min_occurrence, window_size):
        """
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            text_words = f.read().lower().split()
        # Compute the occurences 
        vocab = [('UNK', -1)]
        vocab.extend(collections.Counter(text_words).most_common(max_size-1))
        # Remove the less occurence samples 
        for i in range(len(vocab)-1, -1, -1):
            if vocab[i][1] < min_occurrence:
                vocab.pop(i)
            else:
                break
        vocab_size = len(vocab)
        # Assign id to each word 
        word_to_idx = dict()
        for i, (word, _) in enumerate(vocab):
            word_to_idx[word] = i
        unk_count = 0 
        for word in text_words:
            index = word_to_idx.get(word, 0)
            if index==0:
                unk_count += 1 
        vocab[0] = ('UNK', unk_count)
        # Word pair 
        word_pair = [(vocab[i][0], vocab[i+j][0]) for j in range(-window_size, window_size + 1) if j != 0 \
                        for i in range(window_size, vocab_size-window_size)]
        return word_pair, vocab, word_to_idx, vocab_size


class WIKILitDataModule(plt.LightningDataModule):
    """
    """
    def __init__(self,
            data_path: str='../../data/text8',
            window_size: int=2,
            max_size: int=500000,
            min_occurence: int=10,
            batch_size: int=32,
            num_workers: int=0,
            pin_memory: bool=False
        ):
        super().__init__()
        self.save_hyperparameters()
        self.data_train: Optional[Dataset] = None
        self.prepare_data()

    def prepare_data(self):
        self.data_train = WikiDataset(
            data_path=self.hparams.data_path,
            window_size=self.hparams.window_size,
            max_size=self.hparams.max_size,
            min_occurence=self.hparams.min_occurence
        )

    def train_dataloader(self):
        return DataLoader(
            dataset = self.data_train,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers,
            pin_memory = self.hparams.pin_memory, 
            shuffle = True
        )


class WordEmbedding(nn.Module):
    """
    """
    def __init__(self, vocab_size: int=500000, embedding_size: int=200):
        """
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        """
        """
        x = self.embeddings(x) # [batch, embedding_size]
        x = self.linear(x) # [batch, vocab_size]
        return F.log_softmax(x, dim=1) # [batch, 1]


class LitWordEmbedding(plt.LightningModule):
    """
    """
    def __init__(self, 
            vocab_size: int=500000, 
            embedding_size: int=200,
            learning_rate: float=0.01
        ):
        """
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = WordEmbedding(vocab_size, embedding_size)
        self.criterion = nn.NLLLoss()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)



def main():
    """
    """
    # Hyperparameters 
    window_size = 2
    max_vocabulary_size = 500000 # Total number of different words in the vocabulary.
    embedding_size = 50
    min_occurence = 10
    learning_rate = 0.01
    epochs = 5
    batch_size = 800
    result_dir = 'model'

    # Dataset 
    wiki_data = WIKILitDataModule(
        data_path='../../data/text8',
        window_size=window_size,
        max_size=max_vocabulary_size,
        min_occurence=min_occurence,
        batch_size=batch_size
    )

    # Model 
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=result_dir,
        filename="model-{epoch:02d}-{train_loss:.2f}",
        save_top_k=5,
        mode="min",
    )
    vocab_size = wiki_data.data_train.vocab_size
    model = LitWordEmbedding(vocab_size, embedding_size, learning_rate)

    # Training 
    trainer = plt.Trainer(
        gpus=None, 
        max_epochs=epochs,
        progress_bar_refresh_rate=20,
        callbacks=[checkpoint_callback]
    )
    result = trainer.fit(model, wiki_data)


if __name__ == '__main__':
    """
    """
    main()