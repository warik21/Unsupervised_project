import torch
import torch.nn as nn
from tqdm import tqdm
import time
from torch.utils.data import TensorDataset, DataLoader
import os

class Autoencoder(nn.Module):
    def __init__(self, n_features: int):
        super(Autoencoder, self).__init__()
        self.weight_path: str = r'C:\Users\eriki\Documents\school\Unsupervised_learning\Final_Project\aeacus\weights\ae.pth'
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

    def train_model(self, train_X, train_y, test_X, test_y, optimizer, criterion=nn.MSELoss(), num_epochs=50):
        """
        Trains the specified model.

        Args:
            self: A PyTorch model to be trained.
            train_X: A tensor representing the training data.
            train_y: A tensor representing the training labels.
            test_X: A tensor representing the test data.
            test_y: A tensor representing the test labels.
            criterion: The loss function used for training.
            optimizer: The optimization algorithm used for training.
            num_epochs: An integer representing the number of training epochs.

        Returns:
            A list containing the training loss for each epoch.
        """
        # Check if a GPU is available and move the model to the device
        start = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Create Dataloaders for both the train and test sets
        train_dataset = TensorDataset(train_X, train_y)
        train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        test_dataset = TensorDataset(test_X, test_y)
        test_data_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

        # Initialize a list to store the training losses for each epoch
        train_losses = []
        val_losses = []

        # Compute the lengths once to save runtime
        train_length = len(train_data_loader.dataset)
        test_length = len(test_data_loader.dataset)

        # Loop over the specified number of epochs
        for epoch in tqdm(range(num_epochs)):
            # Initialize the training loss for the current epoch
            train_loss = 0.0
            test_loss = 0.0

            # Loop over the batches of training data
            for inputs, labels in train_data_loader:
                # Zero the gradients
                optimizer.zero_grad()

                # Move the inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass through the model
                outputs = self(inputs)

                # Compute the loss
                loss = criterion(outputs, inputs)

                # Backward pass through the model and optimizer step
                loss.backward()
                optimizer.step()

                # Add the batch loss to the total epoch loss
                train_loss += loss.item()

            # Compute the average epoch loss and add it to the list of losses
            train_loss /= train_length
            train_losses.append(train_loss)

            val_loss = 0
            val_losses.append(val_loss / test_length)

            print('Epoch {}, Loss  {} - Val_loss: {}'.format(epoch, train_loss, test_loss))

        end = time.time()
        training_time = end - start

        # Save the model weights
        self.save_weight()

        # Return the list of training losses
        return train_losses, val_losses, self, training_time

    def load_weight(self):
        if os.path.exists(self.weight_path):
            state_dict = torch.load(self.weight_path)
            self.load_state_dict(state_dict)
            print("found saved weight.")
        else:
            print("no saved weight found.")

    def save_weight(self):
        torch.save(self.state_dict(), self.weight_path)
        print("weight saved.")