import numpy as np
import pandas as pd
import torch

from aeacus.src.utils.autoencoder import Autoencoder


def generate_augmented_samples(fraud_df: pd.DataFrame, model: Autoencoder, num_samples: int) -> pd.DataFrame:
    """
    Trains the specified self on the specified data and labels.

    Args:
        fraud_df: The dataframe containing the original data
        model: The vae model
        num_samples: The number of samples we want to generate

    Returns:
        A dataframe with all the new generated samples
    """
    new_samples = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fraud_df_subset = torch.tensor(fraud_df.values, dtype = torch.float32)
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

class DataHandler:
    def __init__(self, X: pd.DataFrame, y: np.array):
        self.X: pd.DataFrame = X
        self.y: np.array = y


def enrich_data(original_dataframe: pd.DataFrame, auto_encoder: Autoencoder, n_samples: int) -> DataHandler:
    original_dataframe = original_dataframe.drop('Time', axis=1)
    fraud_df_subset = original_dataframe[original_dataframe['Class'] == 1].drop('Class', axis=1)
    original_y = original_dataframe[original_dataframe.columns[29]].to_numpy()
    original_dataframe = original_dataframe.drop('Class', axis=1)

    added_samples: pd.DataFrame = generate_augmented_samples(fraud_df_subset, auto_encoder, n_samples)
    added_samples.columns = original_dataframe.columns
    added_y: np.array = np.ones(len(added_samples))
    enriched_data: pd.DataFrame = pd.concat([original_dataframe, added_samples])
    enriched_data.reset_index(drop=True, inplace=True)
    enriched_y: np.array = np.concatenate((original_y, added_y), axis=0)

    return DataHandler(enriched_data, enriched_y)