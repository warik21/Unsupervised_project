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


class Classifier1(nn.Module):
    """
    A PyTorch module class that defines a self neural network.

    The network consists of three fully connected layers with ReLU activations
    followed by a sigmoid activation in the final layer. The input to the network
    is a tensor of shape (batch_size, 5) and the output is a tensor of shape (batch_size, 1).
    """

    def __init__(self):
        """
        Constructor for the Classifier class.

        Initializes the layers of the network using the nn.Sequential container.
        """
        super(Classifier1, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the self neural network.
        Args:
            x: A tensor of shape (batch_size, 5) representing the input to the network.
        Returns:
            A tensor of shape (batch_size, 1) representing the output of the network.
        """
        x = self.layers(x)
        return x

    def train(self, encoder, train_data, train_labels, val_data, val_labels, optimizer,
              criterion=nn.BCELoss(), num_epochs=2):
        """
        Trains the specified self on the specified data and labels.

        Args:
            self: A PyTorch self to be trained.
            encoder: A trained encoder which reduces the data to the latent dimension.
            train_data: A tensor representing the training data.
            train_labels: A tensor representing the training labels.
            val_data: A tensor representing the validation data.
            val_labels: A tensor representing the validation labels.
            criterion: The loss function used for training.
            optimizer: The optimization algorithm used for training.
            num_epochs: An integer representing the number of training epochs.

        Returns:
            A list containing the training loss for each epoch.
        """
        # Check if a GPU is available and move the self to the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Create a DataLoader object to load the training data in batches
        dataset = TensorDataset(train_data, train_labels)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        val_dataset = TensorDataset(val_data, val_labels)
        val_data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        # Initialize a list to store the training losses for each epoch
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # Loop over the specified number of epochs
        for epoch in tqdm(range(num_epochs)):
            # Initialize the training loss for the current epoch
            train_loss = 0.0
            val_loss = 0.0
            num_correct_train = 0
            num_examples_train = 0
            # Loop over the batches of training data
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                latent_vectors = encoder(inputs)

                # Zero the gradients
                optimizer.zero_grad()

                # Move the inputs and labels to the device
                latent_vectors = latent_vectors.to(device)
                labels = labels.to(device)

                # Forward pass through the self
                outputs = self(latent_vectors)
                outputs = outputs.squeeze()

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass through the self and optimizer step
                loss.backward()
                optimizer.step()

                # Add the batch loss to the total epoch loss
                train_loss += loss.item()

                # Compute the accuracy
                preds = torch.round(outputs)

                # Update the number of correct predictions and total examples
                num_correct_train += torch.sum(preds == labels).item()
                num_examples_train += labels.shape[0]

            # Compute the average epoch loss and add it to the list of losses
            train_loss /= len(data_loader.dataset)
            train_losses.append(train_loss)
            train_accuracy = num_correct_train / num_examples_train
            train_accuracies.append(train_accuracy)

            # Loop over the batches of the validation
            num_correct_val = 0
            num_examples_val = 0

            # Loop over the batches of Validation data
            with torch.no_grad():
                for inputs, labels in val_data_loader:
                    inputs = inputs.to(device)
                    latent_vectors = encoder(inputs)
                    # Move the inputs and labels to the device
                    latent_vectors = latent_vectors.to(device)
                    labels = labels.to(device)

                    # Forward pass through the self
                    outputs = self(latent_vectors)
                    outputs = outputs.squeeze()

                    # Add the batch loss to the total epoch loss
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Compute the predicted labels
                    preds = torch.round(outputs)

                    # Update the number of correct predictions and total examples
                    num_correct_val += torch.sum(preds == labels).item()
                    num_examples_val += labels.shape[0]
            val_loss /= len(val_data_loader.dataset)
            val_losses.append(val_loss)
            val_accuracy = num_correct_val / num_examples_val
            val_accuracies.append(val_accuracy)

            print('Epoch {}, Loss  {} - Accuracy: {} - Val_loss: {} - Val_accuracy: {}'.format(epoch, train_loss,
                                                                                               train_accuracy, val_loss,
                                                                                               val_accuracy))
            # if (epoch + 1) % 20 == 0:
            #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1,
            #                                                num_epochs, train_loss / len(train_data)))

        # Return the list of training losses
        return train_losses, train_accuracies, val_losses, val_accuracies, self

