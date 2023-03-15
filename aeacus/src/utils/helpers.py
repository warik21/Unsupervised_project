import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import torch.optim as optim
import numpy as np

from aeacus.src.utils.classifiers import SVDD, Classifier
from aeacus.src.utils.data_generator import run_through_vae, DataHandler, enrich_data, enrich_data_smote
from aeacus.src.utils.evaluation import TrainingMetrics
from aeacus.src.utils.pre_proccess import prepare_data
from aeacus.src.utils.autoencoder import Autoencoder


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

def train_vae(train_X, train_y, test_X, test_y):
    training_set_feature_count: int = train_X.shape[1]
    auto_encoder: Autoencoder = Autoencoder(n_features=training_set_feature_count)
    optimizer: optim.Adam = optim.Adam(auto_encoder.parameters(), lr=0.001)
    train_losses, test_losses, trained_model, train_model_time = \
        auto_encoder.train_model(train_X, train_y, test_X, test_y, optimizer)
    return train_losses, test_losses, trained_model, train_model_time

def suggested_approach(data_frame: pd.DataFrame, auto_encoder: Autoencoder, device):
    data_frame_good_samples: pd.DataFrame = data_frame[data_frame['Class'] == 0]
    train_ae_svdd_X, test_ae_svdd_X_without_fraud, train_ae_svdd_y, test_ae_svdd_y_without_fraud = prepare_data(data_frame_good_samples)
    train_ae_svdd_X = run_through_vae(train_ae_svdd_X, auto_encoder, device)
    test_classifier = SVDD(nu=0.1, input_dim=29)
    test_classifier = test_classifier.train(train_ae_svdd_X)
    test_ae_svdd_X_without_fraud = run_through_vae(test_ae_svdd_X_without_fraud, auto_encoder, device)

    ae_svdd_model_params = test_classifier.eval_model(data_frame, test_ae_svdd_X_without_fraud, test_ae_svdd_y_without_fraud)

    return test_classifier, ae_svdd_model_params

def latent_enrichment_approach(data_frame: pd.DataFrame, auto_encoder: Autoencoder,
                               n_samples: int, test_X: torch.tensor, test_y: torch.tensor):
    enriched_data: DataHandler = enrich_data(data_frame, auto_encoder, n_samples)
    enriched_classifier: Classifier = Classifier('latent')
    enriched_optimizer: optim.Adam = optim.Adam(enriched_classifier.parameters(), lr=0.001)
    enriched_data_frame: pd.DataFrame = pd.concat(
        [enriched_data.X, pd.DataFrame(np.array(enriched_data.y).reshape(-1, 1))], axis=1)
    enriched_train_X, enriched_test_X, enriched_train_y, enriched_test_y = prepare_data(enriched_data_frame)
    # enriched_train_X = run_through_vae(enriched_train_X, auto_encoder, device)
    enriched_training_metrics: TrainingMetrics = \
        enriched_classifier.train_model(auto_encoder.encoder, enriched_train_X, enriched_train_y,
                                        test_X, test_y, enriched_optimizer)

    # enriched_test_X = run_through_vae(enriched_test_X, auto_encoder, device)
    enriched_classifier_params = enriched_classifier.eval_model(auto_encoder, enriched_test_X, enriched_test_y)

    return enriched_classifier, enriched_classifier_params

def latent_enrichment_approach_with_vae(data_frame: pd.DataFrame, auto_encoder: Autoencoder,
                               n_samples: int, test_X: torch.tensor, test_y: torch.tensor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enriched_data: DataHandler = enrich_data(data_frame, auto_encoder, n_samples)
    enriched_classifier: Classifier = Classifier('latent_with_vae')
    enriched_optimizer: optim.Adam = optim.Adam(enriched_classifier.parameters(), lr=0.001)
    enriched_data_frame: pd.DataFrame = pd.concat(
        [enriched_data.X, pd.DataFrame(np.array(enriched_data.y).reshape(-1, 1))], axis=1)
    enriched_train_X, enriched_test_X, enriched_train_y, enriched_test_y = prepare_data(enriched_data_frame)
    enriched_train_X = run_through_vae(enriched_train_X, auto_encoder, device)
    enriched_training_metrics: TrainingMetrics = \
        enriched_classifier.train_model(auto_encoder.encoder, enriched_train_X, enriched_train_y,
                                        test_X, test_y, enriched_optimizer)

    enriched_test_X = run_through_vae(enriched_test_X, auto_encoder, device)
    enriched_classifier_params = enriched_classifier.eval_model(auto_encoder, enriched_test_X, enriched_test_y)

    return enriched_classifier, enriched_classifier_params
def smote_enrichment_approach(data_frame: pd.DataFrame, auto_encoder: Autoencoder,
                               test_X: torch.tensor, test_y: torch.tensor):
    enriched_data_smote: DataHandler = enrich_data_smote(data_frame)
    enriched_smote_classifier = Classifier('smote')
    enriched_smote_optimizer: optim.Adam = optim.Adam(enriched_smote_classifier.parameters(), lr=0.001)
    enriched_data_frame: pd.DataFrame = pd.concat([enriched_data_smote.X, pd.DataFrame(np.array(enriched_data_smote.y).reshape(-1,1))], axis=1)
    enriched_smote_train_X, enriched_smote_test_X, enriched_smote_train_y, enriched_smote_test_y = prepare_data(enriched_data_frame)
    # enriched_smote_train_X = run_through_vae(enriched_smote_train_X, auto_encoder, device)
    enriched_smote_training_metrics: TrainingMetrics = \
        enriched_smote_classifier.train_model(auto_encoder.encoder, enriched_smote_train_X, enriched_smote_train_y,
                                              test_X, test_y, enriched_smote_optimizer)

    enriched_smote_params = enriched_smote_classifier.eval_model(auto_encoder, enriched_smote_test_X,
                                                                 enriched_smote_test_y)
    return enriched_smote_classifier, enriched_smote_params

def smote_enrichment_approach_with_vae(data_frame: pd.DataFrame, auto_encoder: Autoencoder,
                               test_X: torch.tensor, test_y: torch.tensor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enriched_data_smote: DataHandler = enrich_data_smote(data_frame)
    enriched_smote_classifier = Classifier('smote_with_vae')
    enriched_smote_optimizer: optim.Adam = optim.Adam(enriched_smote_classifier.parameters(), lr=0.001)
    enriched_data_frame: pd.DataFrame = pd.concat([enriched_data_smote.X, pd.DataFrame(np.array(enriched_data_smote.y).reshape(-1,1))], axis=1)
    enriched_smote_train_X, enriched_smote_test_X, enriched_smote_train_y, enriched_smote_test_y = prepare_data(enriched_data_frame)
    enriched_smote_train_X = run_through_vae(enriched_smote_train_X, auto_encoder, device)
    enriched_smote_training_metrics: TrainingMetrics = \
        enriched_smote_classifier.train_model(auto_encoder.encoder, enriched_smote_train_X, enriched_smote_train_y,
                                              test_X, test_y, enriched_smote_optimizer)

    enriched_smote_params = enriched_smote_classifier.eval_model(auto_encoder, enriched_smote_test_X,
                                                                 enriched_smote_test_y)
    return enriched_smote_classifier, enriched_smote_params
