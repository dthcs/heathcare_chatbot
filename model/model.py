import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchsummary import summary

from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from tqdm.notebook import tqdm
import torch.optim as optim
from pathlib import Path
if __name__ == '__main__':

    p_train = Path('dataset/mitbih_train.csv')
    p_test = Path('dataset/mitbih_test.csv')
    p_normal = Path('dataset/ptbdb_normal.csv')
    p_abnormal = Path('dataset/ptbdb_abnormal.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device')
    train_df = pd.read_csv(p_train, header = None )
    test_df = pd.read_csv(p_test, header = None )

    X_train = train_df.loc[:, :186]
    y_train = np.array(train_df.loc[:, 187:])

    X_test = test_df.loc[:, :186]
    y_test = np.array(test_df.loc[:, 187:])

    X_train = X_train.apply(lambda x: Image.fromarray(x.values.reshape(11, 17), 'L'), axis=1)
    X_test = X_test.apply(lambda x: Image.fromarray(x.values.reshape(11, 17), 'L'), axis=1)

    y_train = np.array(y_train).astype(int).squeeze()
    y_test = np.array(y_test).astype(int).squeeze()

    class ImageDataset(Dataset):
        def __init__(self, images_series, labels, transform=None):
            self.images_series = images_series
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images_series)

        def __getitem__(self, idx):
            image = self.images_series.iloc[idx]

            if self.transform:
                image = self.transform(image)

            label = self.labels[idx]

            return image, label

    img_size = 112
    lr = 0.001
    num_epochs = 24
    batch_size = 256


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Updated Data Augmentations
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    # Data augmentation transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


    # Assuming you have a custom ImageDataset class, you can create train and test datasets like this:
    train_dataset = ImageDataset(X_train, y_train, transform=train_transform)
    test_dataset = ImageDataset(X_test, y_test, transform=test_transform)

    # Use a smaller batch size to reduce GPU memory usage
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False  # No need to shuffle test data
    )

    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)

    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.features[-1].fc = nn.AdaptiveAvgPool2d(output_size=1)
    model.avgpool = nn.Identity()
    model.classifier.fc = nn.Linear(1000, 512)
    model.classifier.fc1 = nn.Linear(512, 128)
    model.classifier.fc2 = nn.Linear(128, 5)
    model = model.to('cuda')

    print(device)

    from torch.utils.tensorboard import SummaryWriter
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training Data Size : {len(train_loader)}")
    print(f"Validation Data Size : {len(test_loader)}")
    writer = SummaryWriter()
    model.train()
    cnt = 1e9
    for epoch in range(1024):
        for i, (inps, labels) in enumerate(train_loader):
            inps = inps.to('cuda')
            labels = labels.type(torch.LongTensor)
            labels = labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inps)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar("EfficientNet/Loss/train", loss, epoch)
            model.eval()
            with torch.no_grad():
                sum_loss = 0
                for i, (inps, labels) in enumerate(test_loader):
                    inps = inps.to('cuda')
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to('cuda')
                    outputs = model(inps)
                    sum_loss += criterion(outputs, labels)
                if cnt > sum_loss:
                    cnt = sum_loss
                    torch.save(model.state_dict(), 'effnn.pth')
                    print(f'model saved at {epoch + 1}th epoch.')
            model.train()
        if not (epoch % 16):
            print(f'[{epoch + 1}] loss: {loss.item():.3f}')