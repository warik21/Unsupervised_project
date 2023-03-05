import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
