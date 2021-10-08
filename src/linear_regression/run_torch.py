from sklearn.model_selection import train_test_split 
from torch import nn
from torchsummary import summary
import pandas as pd 
import torch 


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


class LinearRegression(nn.Module):
    """
    """
    def __init__(self, input_dim, output_dim):
        """
        """
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)


def main():
    """
    """
    # Hyper parameters:
    batch_size = 64
    epochs = 200 
    learning_rate = 0.01 
    result = 'model'

    # Dataset
    X_train, Y_train, X_test, Y_test = load_data(data_path='../../data/housing.csv')
    X_train, Y_train = torch.tensor(X_train.values, dtype=torch.float), torch.tensor(Y_train.values, dtype=torch.float).reshape(-1, 1)
    X_test, Y_test = torch.tensor(X_test.values, dtype=torch.float), torch.tensor(Y_test.values, dtype=torch.float).reshape(-1, 1)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    # Define model 
    model = LinearRegression(input_dim=input_dim, output_dim=output_dim)
    print(model.named_parameters)
    print(summary(model, input_size=(input_dim, )))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training 
    for epoch in range(epochs):
        train_sum_loss = 0 
        for x, y in train_loader:
            logit = model(x)
            loss = criterion(logit, y)
            train_sum_loss += loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%20==0:
            test_loss = criterion(model(X_test), Y_test)
            print("Epoch {}: Train loss: {} -- Test loss: {}".format(epoch, train_sum_loss/len(train_loader), test_loss))        
        
    # Saving model 
    torch.save(model.state_dict(), result)


if __name__ == '__main__':
    main()
