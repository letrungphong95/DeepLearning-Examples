import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from tensorflow.keras.layers import Layer

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

class CIFAR10DataGenerator(tf.keras.utils.Sequence):
    """
    """
    def __init__(self, 
            classes: List[str]=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
            'frog', 'horse', 'ship', 'truck'],
            data_path: str='../../data/cifar-10', 
            stage: str='train',
            batch_size: int=32,
            input_size: tuple=(32, 32, 3),
            shuffle: bool=True
        ):
        super().__init__()
        self.classes = {v: i for i, v in enumerate(classes)}
        self.num_classes = len(classes)
        self.data_path = Path(data_path) 
        self.stage = stage
        self.batch_size = batch_size
        self.input_size = input_size 
        self.shuffle = shuffle
        self._read_csv()
        
    def _read_csv(self):
        if self.stage == 'train':
            data = pd.read_csv(self.data_path / 'trainLabels.csv')
        elif self.stage == 'test':
            data = pd.read_csv(self.data_path / 'test.csv')
        self.list_names = list(data['id'])
        self.labels = {i: self.classes[v] for i, v in zip(data['id'], data['label'])}

    def __len__(self):
        """This function computes the number of batch an epoch
        """
        return int(np.floor(len(self.list_names) / self.batch_size))

    def __getitem__(self, index: int):
        """This function generates a batch of data
        """
        # get batch name
        batch_names = self.list_names[index*self.batch_size: (index+1)*self.batch_size]
        # generate data 
        x, y = self._data_generation(batch_names)
        return x, y

    def _data_generation(self, batch_names):
        """This function generate X, y for a batch
        x: (batch_size, 32*32*3)
        y: (batch_size, num_classes)
        """
        x = np.empty((self.batch_size, 32*32*3))
        y = np.empty((self.batch_size), dtype=int)
        for i, name in enumerate(batch_names):
            image = tf.keras.preprocessing.image.load_img(self.data_path/self.stage/'{}.png'.format(name))
            image_arr = tf.keras.preprocessing.image.img_to_array(image)
            image_arr = tf.image.resize(image_arr,(self.input_size[0], self.input_size[1])).numpy().reshape(1, -1)
            image_arr = image_arr / 255.0
            x[i,] = image_arr
            y[i,] = self.labels[name]

        return x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batch_names)


class LogisticRegression(Layer):
    """
    """
    def __init__(self, input_dim: int=32*32*3, output_dim: int=10):
        """
        """
        super().__init__()
        self.w = self.add_weight('weight', [input_dim, output_dim])
        self.b = self.add_weight('bias', [output_dim])
    
    def call(self, x):
        x = tf.matmul(x, self.w) + self.b 
        return tf.nn.softmax(x)


def loss_function(y_pred, y_true):
    """Cross entropy loss function
    """
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), 1))


def accuracy(y_pred, y_true):
    """Accuracy
    """
    correction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(correction)


def main():
    """
    """
    # Hyper parameters
    batch_size = 64
    epochs = 5 
    learning_rate = 0.01
    result_dir = 'model'
    h, w, c = 32, 32, 3
    num_classes = 10

    # Dataset
    train_generator = CIFAR10DataGenerator(
            data_path='../../data/cifar-10', 
            stage='train',
            batch_size=batch_size,
            shuffle=True
        )
    test_generator = CIFAR10DataGenerator(
        data_path='../../data/cifar-10', 
        stage='test',
        batch_size=batch_size,
        shuffle=False
    )

    # Model
    model = LogisticRegression(input_dim=h*w*c, output_dim=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Train model 
    for epoch in range(epochs):
        train_sum_loss = 0
        test_sum_loss = 0
        test_sum_acc = 0
        for x, y in train_generator:
            with tf.GradientTape() as tage:
                logit = model(x)
                loss = loss_function(logit, y)
                train_sum_loss += loss
            grads = tage.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch%1 == 0:
            for x_test, y_test in test_generator:
                logit = model(x_test)
                loss = loss_function(logit, y)
                acc = accuracy(logit, y_test)
                test_sum_loss += loss
                test_sum_acc += acc
            print("Epoch {}: Train loss: {} -- Test loss: {} -- Test acc {}".format(epoch, \
                    train_sum_loss/len(train_generator), \
                    test_sum_loss/len(test_generator), \
                    test_sum_acc/len(test_generator)))

    # Saving model 
    model.save(result_dir)
    
if __name__ == '__main__':
    main()
