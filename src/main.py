import pandas as pd
import numpy as np
from os.path import join
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from torch.utils.data import DataLoader
from timm import create_model
from decimal import Decimal

DATA_PATH = 'data'
socal_df = pd.read_csv(join(DATA_PATH, 'socal2.csv'))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, split='train', transform=None):
        self.df = df
        self.X = df.drop(columns=['price', 'image_id'])
        self.img_dir = join(DATA_PATH, 'socal_pics')
        self.y = df['price']
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = join(self.img_dir, str(self.df.iloc[idx]['image_id'])+ '.jpg')
        img = Image.open(img_path)
        img = self.transform(img)

        return {
            'image': img,
            'sqft': torch.tensor(self.X.iloc[idx]['sqft'], dtype=torch.float),
            'price': torch.tensor(self.y.iloc[idx], dtype=torch.float)
        }

train_df, test_df = train_test_split(socal_df, test_size=0.2, random_state=42)
train_df = train_df.head(1024)
test_df = test_df.head(32)


transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

train_dataset = Dataset(train_df, transform=transform)
test_dataset = Dataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = create_model('resnet18', pretrained=True, num_classes=1)
model.fc = torch.nn.Linear(model.fc.in_features, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = model.to(device)

EPOCHS = 10

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for i in range(EPOCHS):
    model.train()
    train_mse = 0
    for batch in train_loader:
        optimizer.zero_grad()
        X, y = batch['image'].to(device), batch['price'].to(device)
        y_pred = model(X).squeeze()
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_mse += loss.item()
    print(f'Epoch {i+1}\nTrain Loss: {Decimal(loss.item()):.2E}')
    print(f'Train MSE: {Decimal(train_mse/len(train_loader)):.2E}')

    model.eval()
    test_mse = 0
    for batch in test_loader:
        X, y = batch['image'].to(device), batch['price'].to(device)
        y_pred = model(X).squeeze()
        loss = criterion(y_pred, y)
        test_mse += loss.item()
    print(f'Test Loss: {Decimal(loss.item()):.2E}')
    print(f'Test MSE: {Decimal(test_mse/len(test_loader)):.2E}')
