from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
import os
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from aeacus.src.utils.evaluation import TrainingMetrics, ModelEvaluators


class Classifier(nn.Module):
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
        super(Classifier, self).__init__()

        self.layers: nn.Sequential = nn.Sequential(
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
              criterion=nn.BCELoss(), num_epochs=20) -> TrainingMetrics:
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
        dataset: TensorDataset = TensorDataset(train_data, train_labels)
        data_loader: DataLoader = DataLoader(dataset, batch_size=64, shuffle=True)

        val_dataset: TensorDataset = TensorDataset(val_data, val_labels)
        val_data_loader: DataLoader = DataLoader(val_dataset, batch_size=64, shuffle=True)
        # Initialize a list to store the training losses for each epoch
        train_losses: List[float] = []
        train_accuracies: List[float] = []
        val_losses: List[float] = []
        val_accuracies: List[float] = []

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
                predictions = torch.round(outputs)

                # Update the number of correct predictions and total examples
                num_correct_train += torch.sum(predictions == labels).item()
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
                    predictions = outputs

                    # Update the number of correct predictions and total examples
                    num_correct_val += torch.sum(predictions == labels).item()
                    num_examples_val += labels.shape[0]
            val_loss /= len(val_data_loader.dataset)
            val_losses.append(val_loss)
            val_accuracy = num_correct_val / num_examples_val
            val_accuracies.append(val_accuracy)

            print('Epoch {}, Loss  {} - Accuracy: {} - Val_loss: {} - Val_accuracy: {}'.format(epoch, train_loss,
                                                                                               train_accuracy, val_loss,
                                                                                               val_accuracy))
        val_data = val_data.to(device)
        predictions = self(encoder(val_data))
        # Return the list of training losses
        return TrainingMetrics(train_losses=train_losses, train_accuracies=train_accuracies,
                               val_losses=val_losses, val_accuracies=val_accuracies,
                               predictions=predictions)

    def eval_model(self, auto_encoder, enriched_X, enriched_y) -> ModelEvaluators:
        enriched_X = enriched_X.to('cuda:0')
        y_pred = self(auto_encoder.encoder(enriched_X))
        y_pred = y_pred.detach().cpu()
        y_pred = y_pred.round()
        y_true = enriched_y
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return ModelEvaluators(
            accuracy=acc, precision=precision, recall=recall, f1=f1)

class SVDD(nn.Module):
    def __init__(self, input_dim, nu=0.1):
        super(SVDD, self).__init__()
        self.input_dim: int = input_dim
        self.weight_path: str = 'weight/SVDD.pth'
        self.center: np.array = None
        self.nu: float = nu
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 22),
            nn.LeakyReLU(),
            nn.Linear(22, 15),
            nn.LeakyReLU(),
            nn.Linear(15, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1)
        )
        self.loss_mse = nn.MSELoss()
        self.optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001, eps=1)

    def forward(self, X):
        out: float = self.layers(X)
        return out

    def calc_cost(self, X):
        with torch.no_grad():
            dist = torch.sum((X - self.center)**2, dim=1)
            score = dist - self.radius**2
            cost = torch.mean(torch.max(torch.tensor([torch.zeros_like(score)], dtype=torch.float32), score))
        return cost.item()

    def predict(self, X, threshold=0.1):
        with torch.no_grad():
            logits = self.forward(X)
            dist = (logits - self.center).pow(2).sum(dim=1)
            pred = dist.le(threshold).float()
        return pred.cpu().numpy()

    def train(self, X, num_epochs=20, batch_size=128):
        # Check if a GPU is available and move the model to the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        X = X.to(device)
        # Normalize the data
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        # Create a DataLoader object to load the data in batches
        dataset = torch.utils.data.TensorDataset(X)


        #initiate a DataLoader
        data_loader: DataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the center of the hypersphere to the mean of the data
        self.center = X.mean(dim=0, keepdim=True).to(device)
        # Compute the radius of the hypersphere based on the target nu value
        num_outliers = math.ceil(self.nu * X.shape[0])
        distances = (self.center - X).pow(2).sum(dim=1)
        radius, _ = torch.topk(distances, num_outliers)
        radius = radius[-1].sqrt()

        # Loop over the specified number of epochs
        for epoch in range(num_epochs):
            # Initialize the training loss for the current epoch
            train_loss = 0.0

            # Loop over the batches of training data
            for inputs in data_loader:
                inputs = inputs[0].to(device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass through the model
                logits = self.forward(inputs)

                # Compute the distance to the center of the hypersphere
                dist = (logits - self.center).pow(2).sum(dim=1)

                # Compute the loss
                loss = torch.mean(torch.where(dist < radius ** 2, dist, radius ** 2))

                # Backward pass through the model and optimizer step
                loss.backward()
                self.optimizer.step()

                # Add the batch loss to the total epoch loss
                train_loss += loss.item()

                # Compute the average epoch loss and print it to the console
            train_loss /= len(data_loader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss}")

        # Update the center of the hypersphere
        with torch.no_grad():
            self.center = X[self.predict(X) > 0].mean(dim=0, keepdim=True).to(device)

        # Save the model weights
        # TODO: implement the save method correctly.
        self.save_weight()

        return

    def eval_model(self, data_frame, test_X_no_fraud, test_y_no_fraud) -> ModelEvaluators:
        data_frame_fraud_samples: pd.DataFrame = data_frame[data_frame['Class'] == 1]

        data_frame_fraud_samples = data_frame_fraud_samples.drop(['Time'], axis=1)
        data_frame_fraud_samples.columns = range(30)
        df_test_ae_svdd_X_without_fraud = pd.DataFrame(test_X_no_fraud.numpy())
        df_test_ae_svdd_y_without_fraud = pd.DataFrame(test_y_no_fraud.numpy())
        df_without_fraud = pd.concat([df_test_ae_svdd_X_without_fraud, df_test_ae_svdd_y_without_fraud], axis=1,
                                     ignore_index=True)
        test_df = pd.concat([df_without_fraud, data_frame_fraud_samples], ignore_index=True)

        X_test = test_df.iloc[:, :-1]
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        X_mean = torch.mean(X_test, dim=0)
        X_std = torch.std(X_test, dim=0)
        X_test = (X_test - X_mean) / X_std

        # make predictions on the test data
        X_test = X_test.to('cuda:0')
        y_pred = self.predict(X_test, threshold=0.1)
        y_true = test_df.iloc[:, -1].values
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return ModelEvaluators(accuracy = acc, precision = precision, recall = recall, f1 = f1)

    def load_weight(self):
        if os.path.exists(self.weight_path):
            state_dict = torch.load(self.weight_path)
            self.load_state_dict(state_dict)
            print("found saved weight.")
        else:
            print("no saved weight found.")

    def save_weight(self):
        # torch.save(self.state_dict(), self.weight_path)
        print("weight saved.")