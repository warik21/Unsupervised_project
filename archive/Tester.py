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
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
import umap

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
            nn.Linear(n_features, 25),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(25, 10),
            nn.ReLU(),

        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 25),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(25, n_features),
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
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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

def prepare_data(df):
    """
    Prepares the data for training the model.

    Args:
        df: The dataframe

    Returns:
        A tuple containing the preprocessed training data and labels.
    """
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

def visualize(df, encoder):
    """
    Visualizes the difference between the latent representations of fraudulent and good transactions
    using principal component analysis (PCA).

    Args:
        df (pd.DataFrame): The DataFrame containing the transaction data, where the last column is the class label.
        encoder (torch.nn.Module): The encoder neural network for the VAE, used to encode the transaction data.

    Returns:
        None (displays the scatter plot using matplotlib)
    """

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Drop 'Time' column and subset data by class label
    temp_df = df.drop('Time', axis=1)
    fraud_df_subset = temp_df[temp_df['Class'] == 1].drop('Class', axis=1)
    good_df_subset = temp_df[temp_df['Class'] == 0].drop('Class', axis=1)
    good_df_subset = good_df_subset.sample(n=len(fraud_df_subset.index), random_state = 42)

    # Convert data to PyTorch tensors and move to GPU (if available)
    fraud_samples = torch.tensor(fraud_df_subset.values, dtype=torch.float32).to(device)
    good_samples = torch.tensor(good_df_subset.values, dtype=torch.float32).to(device)

    # Encode the data using the VAE encoder
    encoded_fraud_samples = encoder(fraud_samples)
    encoded_good_samples = encoder(good_samples)

    # Move encoded data to CPU and convert to NumPy arrays
    encoded_fraud_samples = encoded_fraud_samples.cpu().detach().numpy()
    encoded_good_samples = encoded_good_samples.cpu().detach().numpy()

    # Apply PCA to reduce the dimensionality of the encoded data to 2 dimensions
    pca = PCA(n_components=2)
    fraud_pca = pca.fit_transform(encoded_fraud_samples)
    good_pca = pca.fit_transform(encoded_good_samples)

    # Plot the encoded data using a scatter plot
    plt.scatter(fraud_pca[:, 0], fraud_pca[:, 1], label='Fraud')
    plt.scatter(good_pca[:, 0], good_pca[:, 1], label='Good')
    plt.title('PCA')
    plt.legend()
    plt.show()

def visualize_tsne(df, encoder):
    """
    Visualizes the difference between the data of fraudulent and non-fraudulent transactions in the latent space
    using t-SNE dimensionality reduction technique.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to be visualized.
    encoder : torch.nn.Module
        Encoder model to transform the input data into latent space.

    Returns:
    --------
    None
    """
    # Determine the device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Drop 'Time' column and subset data by class label
    temp_df = df.drop('Time', axis=1)
    fraud_df_subset = temp_df[temp_df['Class'] == 1].drop('Class', axis=1)
    good_df_subset = temp_df[temp_df['Class'] == 0].drop('Class', axis=1)
    good_df_subset = good_df_subset.sample(n=10 * len(fraud_df_subset.index), random_state = 42)

    # Convert data to PyTorch tensors and move to GPU (if available)
    fraud_samples = torch.tensor(fraud_df_subset.values, dtype=torch.float32).to(device)
    good_samples = torch.tensor(good_df_subset.values, dtype=torch.float32).to(device)

    # Encode the data using the VAE encoder
    encoded_fraud_samples = encoder(fraud_samples)
    encoded_good_samples = encoder(good_samples)

    # Move encoded data to CPU and convert to NumPy arrays
    encoded_fraud_samples = encoded_fraud_samples.cpu().detach().numpy()
    encoded_good_samples = encoded_good_samples.cpu().detach().numpy()

    # Create a t-SNE object with 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)

    # Convert the encoded transactions to NumPy arrays and perform t-SNE
    fraud_tsne = tsne.fit_transform(encoded_fraud_samples)
    good_tsne = tsne.fit_transform(encoded_good_samples)

    # Plot the t-SNE results
    plt.scatter(fraud_tsne[:, 0], fraud_tsne[:, 1], label='Fraud')
    plt.scatter(good_tsne[:, 0], good_tsne[:, 1], label='Good')
    plt.title('t-SNE')
    plt.legend()
    plt.show()

def train_model(model, train_X, train_y, test_X, test_y,  optimizer, criterion = nn.MSELoss(),num_epochs = 2):
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
    print('Epoch {}, Loss  {} - Val_loss: {}'.format(epoch, train_loss, test_loss))
    # Return the list of training losses
    return train_losses, val_losses, model, training_time

def generate_augmented_samples(original_df, model, num_samples):
    """
    Trains the specified classifier on the specified data and labels.

    Args:
        original_df: The dataframe containing the original data
        model: The vae model
        num_samples: The number of samples we want to generate

    Returns:
        A dataframe with all the new generated samples
    """
    new_samples = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fraud_df_subset = original_df[original_df['Class'] == 1].drop('Class', axis=1)
    fraud_df_subset = fraud_df_subset.drop('Time', axis = 1)

    fraud_df_subset = torch.tensor(fraud_df_subset.values, dtype = torch.float32)
    fraud_df_subset = fraud_df_subset.to(device)

    #Encode the fraudulent samples
    encoded_fraud = model.encoder(fraud_df_subset)
    #Find the mean and std
    mean_latent_vector = encoded_fraud.mean(dim=0)
    std_latent_vector = encoded_fraud.std(dim=0)

    for i in range(num_samples):
        random_noise = torch.randn_like(mean_latent_vector)
        latent_vector = mean_latent_vector + random_noise * std_latent_vector
        new_sample = model.decoder(latent_vector)
        new_sample = new_sample.detach().cpu()
        new_sample = np.array(new_sample)
        new_samples.append(new_sample)

    df_augmented = pd.DataFrame(new_samples)
    return df_augmented

def train_classifier(classifier, encoder, train_data, train_labels, val_data, val_labels, optimizer, criterion = nn.BCELoss(), num_epochs = 50):
    """
    Trains the specified classifier on the specified data and labels.

    Args:
        classifier: A PyTorch classifier to be trained.
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
    # Check if a GPU is available and move the classifier to the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

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

                # Forward pass through the classifier
                outputs = classifier(latent_vectors)
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


        print('Epoch {}, Loss  {} - Accuracy: {} - Val_loss: {} - Val_accuracy: {}'.format(epoch, train_loss, train_accuracy, val_loss, val_accuracy))
        # if (epoch + 1) % 20 == 0:
        #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1,
        #                                                num_epochs, train_loss / len(train_data)))

    # Return the list of training losses
    return train_losses, train_accuracies, val_losses, val_accuracies, classifier

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
    classifier_losses, classifier_accuracies, test_losses, test_accuracies, classifier = train_classifier(classifier, encoder, train_data, train_labels,
                                                                                                          test_data, test_labels, optimizer)
    # classifier_accuracy = eval_classifier(classifier, encoder, test_data, test_labels)
    end_classifier = time.time()
    classifier_time = end_classifier - start_classifier
    return classifier_losses, classifier_accuracies, test_losses, test_accuracies, classifier_time

def plot_auto_encoder_losses(train_losses, val_losses):
    """
    Takes training and validation losses and plots them together.

    Args:
        train_losses: The losses on the training set.
        val_losses: The losses on the validation set.

    Returns:
        Nothing, prints a plot
    """
    plt.plot(train_losses, 'b')
    plt.plot(val_losses, 'r')
    plt.title('Autoencoer model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['Train_loss', 'Val_loss'], loc='upper right');
    plt.show()

def plot_classifier_accuracies(accuracies_1, accuracies_2, title):
    """
    Plots train and evaluation accuracies of a classifier.

    Parameters:
        - accuracies_1 (set of floats): first set of accuracies of a classifier.
        - accuracies_2 (set of floats): second set of accuracies of a classifier.
        - title (str): Title of the plot.

    Returns:
        None
    """
    plt.plot(accuracies_1, label="Train Accuracies")
    plt.plot(accuracies_2, label="Eval Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    start = time.time()

    data_path = r'C:\Users\eriki\Documents\school\Unsupervised_learning\big_files\creditcard.csv'
    data_frame = pd.read_csv(data_path)
    train_X, test_X, train_y, test_y = prepare_data(data_frame)

    auto_encoder = Autoencoder(n_features=train_X.shape[1])

    optimizer = optim.Adam(auto_encoder.parameters(), lr=0.001)
    train_losses0, test_losses0, trained_model, train_model_time = train_model(auto_encoder, train_X, train_y, test_X, test_y, optimizer)

    generate_augmented_samples(data_frame, auto_encoder, 400)

    encoder = trained_model.encoder
    visualize(data_frame, encoder)
    visualize_tsne(data_frame, encoder)

    aug_data_path = r'C:\Users\eriki\Documents\school\Unsupervised_learning\big_files\rand.csv'
    aug_data_frame = pd.read_csv(aug_data_path)
    aug_train_X, aug_test_X, aug_train_y, aug_test_y = prepare_data(aug_data_frame)

    og_classifier_losses, og_classifier_accuracies, og_eval_losses, og_eval_accuracies, og_classifier_time = time_classifier_train_eval(encoder, train_X, train_y, test_X, test_y)

    aug_classifier_losses, aug_classifier_accuracies, aug_eval_losses, aug_eval_accuracies, aug_classifier_time = time_classifier_train_eval(encoder, aug_train_X, aug_train_y
                                                                              , test_X, test_y)

    end = time.time()
    seconds = end - start
    print('The process took {} seconds, of which {} went to training the autoencoder, {} went to training the first classifier, '
          'and {} to train the second classifier'.format(seconds, train_model_time, og_classifier_time, aug_classifier_time))


    plot_auto_encoder_losses(train_losses0, test_losses0)

    plot_classifier_accuracies(og_classifier_accuracies, og_eval_accuracies, "og train vs eval")
    plot_classifier_accuracies(aug_classifier_accuracies, aug_eval_accuracies, "aug train vs eval")
    plot_classifier_accuracies(og_eval_accuracies, aug_eval_accuracies, "og train vs aug train")



if __name__ == '__main__':
    main()