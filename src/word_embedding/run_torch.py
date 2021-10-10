from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
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
    result_dir = 'model/model.pth'

    # Dataset 
    train_dataset = WikiDataset(
        data_path='../../data/text8',
        window_size=window_size,
        max_size=max_vocabulary_size,
        min_occurence=min_occurence
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # Define model 
    is_gpu = torch.cuda.is_available()
    model = WordEmbedding(train_dataset.vocab_size, embedding_size)
    print(model.named_parameters)
    # print(summary(model, input_size=(1, )))
    if is_gpu:
        model.cuda()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_sum_loss = 0
        model.train()
        for x, y in tqdm(train_loader):
            if is_gpu:
                x.cuda()
                y.cuda()
            model.zero_grad()
            # Compute output
            logit = model(x)
            loss = criterion(logit, y)
            train_sum_loss += loss.item()
            
            # Back propagation 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch {}: {}'.format(epoch, loss.item()))

    # Saving model 
    torch.save(model.state_dict(), result_dir)

if __name__ == '__main__':
    """
    """
    main()
