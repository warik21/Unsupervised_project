import numpy as np
import pandas as pd
import torch


def generate_augmented_samples(original_df, model, num_samples):
    """
    Trains the specified self on the specified data and labels.

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
