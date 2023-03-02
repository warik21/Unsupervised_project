import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets, decomposition, preprocessing
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from torch.utils.data import TensorDataset, DataLoader
# Define the model
class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 20),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(20, n_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



start = time.time()
df = pd.read_csv('creditcard.csv')

x = df[df.columns[1:30]].to_numpy()
y = df[df.columns[30]].to_numpy()
device = torch.device("cuda:0")
print(torch.cuda.is_initialized())

# prepare data
df = pd.concat([pd.DataFrame(x), pd.DataFrame({'anomaly': y})], axis=1)

# Scaling
df = df.drop('anomaly', axis=1)

# 80% of the data for training
train_data, test_data, train_y, test_y = train_test_split(df, y,test_size=0.2)

scaler = preprocessing.MinMaxScaler()
scaler.fit(train_data)
scaled_data = scaler.transform(train_data)

# Convert numpy arrays to tensors
train_data = torch.tensor(scaled_data, dtype=torch.float32)
train_data = train_data.to(device)
train_y = torch.tensor(train_y, dtype = torch.float32)
train_y = train_y.to(device)

n_features = x.shape[1]

autoencoder = Autoencoder(n_features)
autoencoder.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train the model
num_epochs = 20
vae_losses = []

dataset = TensorDataset(train_data, train_y)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in tqdm(range(num_epochs)):
    train_loss = 0.0

    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    vae_losses.append(train_loss / len(train_data))
    if (epoch + 1) % 20 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1,
                                                   num_epochs, train_loss / len(train_data)))


df2_path = r'C:\Users\eriki\Documents\school\Unsupervised_learning\Final_Project\rand.csv'
df_path = r'C:\Users\eriki\Documents\school\Unsupervised_learning\Final_Project\archive\creditcard.csv'

df = pd.read_csv(df_path)
df2 = pd.read_csv(df2_path)

df2 = df2.drop('Unnamed: 0', axis = 1)

scaler = preprocessing.MinMaxScaler()
scaler.fit(df2)
scaled_df2 = scaler.transform(df2)

classifier = nn.Sequential(
    nn.Linear(5, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
classifier = classifier.to(device)

encoder = autoencoder.encoder

df2 = df2.fillna(1)


X = df2.drop('class', axis = 1)
X = torch.tensor(X.values, dtype = torch.float32)
y = torch.tensor(df2['class'].values, dtype = torch.float32)
X = X.to(device)
y = y.to(device)

dataset = TensorDataset(X,y)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

losses = []
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
for epoch in tqdm(range(100)):
    epoch_loss = 0
    for inputs, labels in data_loader:
        # Get the latent vectors from the VAE encoder
        latent_vectors = autoencoder.encoder(inputs).detach()

        # Pass the latent vectors through the classifier network
        predictions = classifier(latent_vectors)

        # Calculate the loss
        labels = labels.view(-1, 1)
        loss = criterion(predictions, labels)
        epoch_loss += loss.item()
        # Backpropagate the error and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(epoch_loss / len(data_loader))

scaler2 = preprocessing.MinMaxScaler()
scaler2.fit(df)
scaled_df = scaler2.transform(df)

X0 = df.drop('Time', axis = 1)
X0 = X0.drop('Class', axis = 1)
X0= torch.tensor(X0.values, dtype = torch.float32)
y0 = torch.tensor(df['Class'].values, dtype = torch.float32)
X0 = X0.to(device)
y0 = y0.to(device)

dataset = TensorDataset(X0,y0)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

losses0 = []
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
for epoch in tqdm(range(100)):
    epoch_loss = 0
    for inputs, labels in data_loader:
        # Get the latent vectors from the VAE encoder
        latent_vectors = autoencoder.encoder(inputs).detach()

        # Pass the latent vectors through the classifier network
        predictions = classifier(latent_vectors)

        # Calculate the loss
        labels = labels.view(-1, 1)
        loss = criterion(predictions, labels)
        epoch_loss += loss.item()
        # Backpropagate the error and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses0.append(epoch_loss / len(data_loader))

end = time.time()
seconds = end - start
print(f'This took a while, about %d', seconds)
