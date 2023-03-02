import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from torch.utils.data import TensorDataset, DataLoader

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, n_features):
        """
        Initializes the Autoencoder model.

        Args:
            n_features: An integer representing the number of features in the input data.
        """
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
        """
        Defines the forward pass of the Autoencoder model.

        Args:
            x: A tensor representing the input data.

        Returns:
            A tensor representing the output of the model.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Classifier1(nn.Module):
    """
    A PyTorch module class that defines a classifier neural network.

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
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the classifier neural network.
        Args:
            x: A tensor of shape (batch_size, 5) representing the input to the network.
        Returns:
            A tensor of shape (batch_size, 1) representing the output of the network.
        """
        x = self.layers(x)
        return x


def prepare_data(df_path):
    """
    Prepares the data for training the model.

    Args:
        df_path: the path to the dataframe

    Returns:
        A tuple containing the preprocessed training data and labels.
    """
    df = pd.read_csv(df_path)
    df = df.fillna(1)
    x = df[df.columns[1:30]].to_numpy()
    y = df[df.columns[30]].to_numpy()

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

    scaler_train = preprocessing.MinMaxScaler()
    scaler_train.fit(train_x)
    scaled_x_train = scaler_train.transform(train_x)

    # Convert numpy arrays to tensors
    train_x = torch.tensor(scaled_x_train, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    scaler_test = preprocessing.MinMaxScaler()
    scaler_test.fit(test_x)
    scaled_x_test = scaler_test.transform(test_x)

    test_x = torch.tensor(scaled_x_test, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    return train_x, test_x, train_y, test_y

def train_model(model, train_X, train_y, test_X, test_y,  optimizer, criterion = nn.MSELoss(),num_epochs = 1):
    """
    Trains the specified model.

    Args:
        model: A PyTorch model to be trained.
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
    model.to(device)

    # Create Dataloaders for both the train and test sets
    train_dataset = TensorDataset(train_X, train_y)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(test_X, test_y)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

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
            outputs = model(inputs)

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

        if (epoch + 1) % 20 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1,
                                                       num_epochs, train_loss / len(train_X)))

        model.eval()
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            # Loop over the batches of validation data
            for inputs, labels in test_data_loader:
                # Move the inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass through the model
                outputs = model(inputs)

                # Compute the validation loss
                loss = criterion(outputs, inputs)

                # Add the batch loss to the total epoch loss
                test_loss += loss.item()

            val_losses.append(test_loss/test_length)

    end = time.time()
    training_time = end - start
    # Return the list of training losses
    return train_losses, val_losses, model, training_time

def train_classifier(classifier, encoder, train_data, train_labels, optimizer, criterion = nn.BCELoss(), num_epochs = 1):
    """
    Trains the specified classifier on the specified data and labels.

    Args:
        classifier: A PyTorch classifier to be trained.
        encoder: A trained encoder which reduces the data to the latent dimension.
        train_data: A tensor representing the training data.
        train_labels: A tensor representing the training labels.
        criterion: The loss function used for training.
        optimizer: The optimization algorithm used for training.
        num_epochs: An integer representing the number of training epochs.

    Returns:
        A list containing the training loss for each epoch.
    """
    # Check if a GPU is available and move the classifier to the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    # Create a DataLoader object to load the training data in batches
    dataset = TensorDataset(train_data, train_labels)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize a list to store the training losses for each epoch
    train_losses = []

    # Loop over the specified number of epochs
    for epoch in tqdm(range(num_epochs)):
        # Initialize the training loss for the current epoch
        train_loss = 0.0

        # Loop over the batches of training data
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            latent_vectors = encoder(inputs)

            # Zero the gradients
            optimizer.zero_grad()

            # Move the inputs and labels to the device
            latent_vectors = latent_vectors.to(device)
            labels = labels.to(device)

            # Forward pass through the classifier
            outputs = classifier(latent_vectors)
            outputs = outputs.squeeze()

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass through the classifier and optimizer step
            loss.backward()
            optimizer.step()

            # Add the batch loss to the total epoch loss
            train_loss += loss.item()

        # Compute the average epoch loss and add it to the list of losses
        train_loss /= len(data_loader.dataset)
        train_losses.append(train_loss)

        if (epoch + 1) % 20 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1,
                                                       num_epochs, train_loss / len(train_data)))

    # Return the list of training losses
    return train_losses, classifier

def eval_classifier(classifier, encoder, test_data, test_labels):
    """
    Evaluates the accuracy of the specified classifier on the specified data and labels.

    Args:
        classifier: A PyTorch classifier to be evaluated.
        encoder: A trained encoder which reduces the data to the latent dimension.
        test_data: A tensor representing the input data.
        test_labels: A tensor representing the true labels.

    Returns:
        The accuracy of the classifier on the specified data and labels.
    """
    # Check if a GPU is available and move the classifier to the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    # Create a DataLoader object to load the data in batches
    dataset = TensorDataset(test_data, test_labels)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Set the classifier to evaluation mode
    classifier.eval()

    # Initialize variables to keep track of the number of correct predictions and total examples
    num_correct = 0
    num_examples = 0

    # Loop over the batches of data
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        latent_vectors = encoder(inputs)
        # Move the inputs and labels to the device
        latent_vectors = latent_vectors.to(device)
        labels = labels.to(device)

        # Forward pass through the classifier
        outputs = classifier(latent_vectors)

        # Compute the predicted labels
        preds = torch.round(outputs)

        # Update the number of correct predictions and total examples
        num_correct += torch.sum(preds == labels).item()
        num_examples += labels.shape[0]

    # Compute the accuracy and return it
    accuracy = num_correct / num_examples
    return accuracy

def time_classifier_train_eval(encoder, train_data, train_labels, test_data, test_labels):
    """
    Takes a classifier and the dataset and returns the accuracy and the time it took to train and evaluate it

    Args:
        encoder: A trained encoder which reduces the data to the latent dimension.
        train_data: A tensor representing the training data.
        train_labels: A tensor representing the training labels.
        test_data: A tensor representing the test data.
        test_labels: A tensor representing the test labels.

    Returns:
        A list containing the training loss for each epoch.
    """
    start_classifier = time.time()
    classifier = Classifier1()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    classifier_losses, classifier = train_classifier(classifier, encoder, train_data, train_labels, optimizer)
    classifier_accuracy = eval_classifier(classifier, encoder, test_data, test_labels)
    end_classifier = time.time()
    classifier_time = end_classifier - start_classifier
    return classifier_accuracy, classifier_time

def main():
    start = time.time()

    data_path = r'C:\Users\eriki\Documents\school\Unsupervised_learning\Final_Project\archive\creditcard.csv'
    train_X, test_X, train_y, test_y = prepare_data(data_path)

    auto_encoder = Autoencoder(n_features=train_X.shape[1])

    optimizer = optim.Adam(auto_encoder.parameters(), lr=0.001)
    train_losses0, test_losses0, trained_model, train_model_time = train_model(auto_encoder, train_X, train_y, test_X, test_y, optimizer)


    encoder = trained_model.encoder

    aug_data_path = r'C:\Users\eriki\Documents\school\Unsupervised_learning\Final_Project\rand.csv'
    aug_train_X, aug_test_X, aug_train_y, aug_test_y = prepare_data(aug_data_path)

    og_classifier_accuracy, og_classifier_time = time_classifier_train_eval(encoder, train_X, train_y, test_X, test_y)

    aug_classifier_accuracy, aug_classifier_time = time_classifier_train_eval(encoder, aug_train_X, aug_train_y
                                                                              , test_X, test_y)

    end = time.time()
    seconds = end - start
    print('The accuracy for the first classifier on the test data is {:.1f}%'.format(og_classifier_accuracy))
    print('The accuracy for the first classifier on the augmented test data is {:.1f}%'.format(aug_classifier_accuracy))
    print('The process took {} seconds, of which {} went to training the autoencoder, {} went to training the first classifier, '
          'and {} to train the second classifier'.format(seconds, train_model_time, og_classifier_time, aug_classifier_time))

    # Visualization of loss fonction
    plt.plot(train_losses0, 'b')
    plt.plot(test_losses0, 'r')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['Train_loss', 'Val_loss'], loc='upper right');
    plt.show()

if __name__ == '__main__':
    main()