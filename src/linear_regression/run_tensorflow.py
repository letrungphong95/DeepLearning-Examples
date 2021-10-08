import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.layers import Layer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class LinearRegression(Layer):
    def __init__(self, feature=13):
        """
        """
        super().__init__()
        # super(LinearRegression, self).__init__()
        self.w = self.add_weight('weight', [feature, 1])
        self.b = self.add_weight('bias', [1])
        print(self.w.shape, type(self.w), self.w.name)
        print(self.w.shape, type(self.w), self.w.name)

    def call(self, x):
        x = tf.matmul(x, self.w) + self.b
        return x 

def load_data(data_path):
    """This function is used to load .csv dataset
    """
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = pd.read_csv(data_path, delimiter=r"\s+", names=column_names)
    Y = data['MEDV']
    X = data.drop('MEDV', axis=1)
    # X = pd.DataFrame(np.c_[data['LSTAT'], data['RM']], columns = ['LSTAT','RM'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    print("Training data: {}, {}".format(X_train.shape, Y_train.shape))
    print("Testing data: {}, {}".format(X_test.shape, Y_test.shape))
    return X_train, Y_train, X_test, Y_test


def main():
    """
    """
    # Hyper parameters
    batch_size = 64
    epochs = 1000 
    learning_rate = 0.01
    result_dir = 'model'
    

    # (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.boston_housing.load_data()
    X_train, Y_train, X_test, Y_test = load_data(data_path='../../data/housing.csv')
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    train_loader = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)
    test_loader = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)
    num_feature = X_train.shape[1]

    # Define model 
    model = LinearRegression(feature=num_feature)
    print(model.summary())
    criteon = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Train model 
    for epoch in range(epochs):
        train_sum_loss = 0
        test_sum_loss = 0
        for x, y in train_loader:
            with tf.GradientTape() as tage:
                logit = model(x) # [batch, 1]
                loss = criteon(tf.squeeze(logit, axis=1), y) # [batch]
                train_sum_loss += loss
            grads = tage.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch%20 == 0:
            for x_test, y_test in test_loader:
                logit = model(x_test)
                loss = criteon(tf.squeeze(logit, axis=1), y_test)
                test_sum_loss += loss
            print("Epoch {}: Train loss: {} -- Test loss: {}".format(epoch, \
                train_sum_loss/len(train_loader), test_sum_loss/len(test_loader)))

    # Saving entire model
    model.save(result_dir)


if __name__ == '__main__':
    main()