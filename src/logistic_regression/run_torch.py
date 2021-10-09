from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from pathlib import Path
from typing import List
from PIL import Image 
from torch import nn
import torchvision.transforms as transforms
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

class LogisticRegression(nn.Module):
    """
    """
    def __init__(self, input_dim: int=32*32*3, output_dim: int=10):
        """
        """
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.model(x)
        return x

def main():
    """
    """
    # Hyper parameters
    batch_size = 32
    epochs = 5 
    learning_rate = 0.01
    result_dir = 'model/model.pth'
    h, w, c = 32, 32, 3
    num_classes = 10

    # Dataset
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    # Train dataset
    train_tranforms = transforms.Compose([
        transforms.RandomResizedCrop(h),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_dataset = CIFAR10Dataset(
        data_path='../../data/cifar-10', 
        stage='train',
        transform=train_tranforms
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Test dataset
    test_tranforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_dataset = CIFAR10Dataset(
        data_path='../../data/cifar-10', 
        stage='test',
        transform=test_tranforms
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model 
    is_gpu = torch.cuda.is_available()
    model = LogisticRegression(input_dim=h*w*c, output_dim=num_classes)
    print(model.named_parameters)
    print(summary(model, input_size=(h*w*c, )))
    if is_gpu:
        model.cuda()
    # computes softmax and then the cross entropy
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training 
    for epoch in range(epochs):
        train_sum_loss = 0 
        train_sum_acc = 0
        test_sum_loss = 0 
        test_sum_acc = 0
        model.train()
        for x, y in train_loader:
            if is_gpu:
                x.cuda()
                y.cuda()
            # Compute output
            logit = model(x)
            loss = criterion(logit, y)
            train_sum_loss += loss.item()
            _, pred = torch.max(logit, 1)
            train_sum_acc += (pred==y).float().mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch%1 == 0:
            model.eval()
            for x_test, y_test in test_loader:
                if is_gpu:
                    x_test.cuda()
                    y_test.cuda()
                with torch.no_grad():
                    logit = model(x_test)
                    try:
                        loss = criterion(logit, y_test)
                    except:
                        print(logit.size, y_test.size)
                    test_sum_loss += loss.item()
                _, pred = torch.max(logit, 1)
                test_sum_acc += (pred==y_test).float().mean()
            print('Epoch {}: Train loss: {} -- Test loss: {} -- Train Acc: {} -- Test Acc: {}'.format(
                epoch, train_sum_loss/len(train_loader), test_sum_loss/len(test_loader),
                train_sum_acc/len(train_loader), test_sum_acc/len(test_loader)
            ))

    # Saving model 
    torch.save(model.state_dict(), result_dir)


if __name__ == '__main__':
    """
    """
    main()