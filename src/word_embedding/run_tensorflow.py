import collections
import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import Layer

class WikiDataGenerator(tf.keras.utils.Sequence):
    """
    """
    def __init__(self,
            data_path: str='../../data/text8',
            batch_size: int=32,
            window_size: int=2,
            max_size: int=500000,
            min_occurrence: int=10,
            shuffle: bool=True
        ):
        """
        """
        super().__init__()
        
        # Definition 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.word_pair, self.vocab, self.word_to_idx, self.vocab_size = self._read_corpus(
            data_path=data_path,
            max_size=max_size,
            min_occurrence=min_occurrence,
            window_size=window_size
        )

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
        

    def __len__(self):
        """
        """
        return len(self.word_pair)

    def __getitem__(self, index: int):
        """
        """
        # get batch idx
        batch_word_pair = self.word_pair[index*self.batch_size: (index+1)*self.batch_size]
        x = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        y = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        for index, word in enumerate(batch_word_pair):
            x[index,] = self.word_to_idx[word[0]]
            y[index, 0] = self.word_to_idx[word[1]]
        return x, tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.word_pair)

class WordEmbedding(Layer):
    """
    """
    def __init__(self, vocab_size: int=None, embedding_size: int=200):
        """
        """
        super().__init__()
        self.embeddings = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.w = self.add_weight('weight', [vocab_size, embedding_size])
        self.b = self.add_weight('bias', [vocab_size])

    def call(self, x):
        """
        """
        x = tf.matmul(self.embeddings(x), tf.transpose(self.w)) + self.b 
        return tf.nn.softmax(x)

def loss_function(y_pred, y_true):
    """Cross entropy loss function
    """
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), 1))


def main():
    """
    """
    # Hyperparameters 
    max_vocabulary_size = 50000 # Total number of different words in the vocabulary.
    embedding_size = 200
    learning_rate = 0.01
    epochs = 5
    batch_size = 32
    result_dir = 'model'
    
    # Dataset 
    train_generator = WikiDataGenerator(
        data_path='../../data/text8',
        batch_size=batch_size)

    # Model
    model = WordEmbedding(vocab_size=train_generator.vocab_size, embedding_size=embedding_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Train model 
    for epoch in range(epochs):
        train_sum_loss = 0
        for x, y in train_generator:
            with tf.GradientTape() as tage:
                logit = model(x)
                loss = loss_function(logit, y)
                train_sum_loss += loss
            grads = tage.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print(logit.shape, y.shape, loss)
        if epoch%1 == 0:
            print("Epoch {}: Train loss: {}".format(epoch, train_sum_loss/len(train_generator)))

    # Saving model 
    model.save(result_dir)


if __name__ == '__main__':
    main()