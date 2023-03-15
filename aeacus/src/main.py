import time
import torch.optim as optim
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

from aeacus.src.utils.data_generator import run_through_vae
from aeacus.src.utils.pre_proccess import prepare_data
from aeacus.src.utils.helpers import suggested_approach, latent_enrichment_approach, smote_enrichment_approach, \
    train_vae, latent_enrichment_approach_with_vae, smote_enrichment_approach_with_vae

if __name__ == '__main__':
    start = time.time()

    # AutoEncoder training pipeline
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path: str = r'C:\Users\eriki\Documents\school\Unsupervised_learning\big_files\creditcard.csv'
    data_frame: pd.DataFrame = pd.read_csv(data_path)
    train_X, test_X, train_y, test_y = prepare_data(data_frame)

    train_losses, test_losses, auto_encoder, train_model_time = train_vae(train_X, train_y, test_X, test_y)

    test_X = run_through_vae(test_X, auto_encoder, device)


    #######################################Article suggested approach #################################################
    ae_svdd_model, ae_svdd_model_params = suggested_approach(data_frame, auto_encoder, device)
    ###################################################################################################################

    #Data enrichment overall parameters:
    n_samples = int(len(data_frame.index) / 2)

    ############################ Using latent space into classifier ####################################################
    enriched_classifier_model, enriched_classifier_params = latent_enrichment_approach(data_frame, auto_encoder, n_samples, test_X, test_y)
    enriched_classifier_with_vae_model, enriched_classifier_params_with_vae = \
        latent_enrichment_approach_with_vae(data_frame, auto_encoder, n_samples, test_X, test_y)
    ###################################################################################################################

    ################################# Using SMOTE #####################################################################
    enriched_smote_model, enriched_smote_params = smote_enrichment_approach(data_frame, auto_encoder, test_X, test_y)
    enriched_smote_with_vae_model, enriched_smote_params_with_vae = \
        smote_enrichment_approach_with_vae(data_frame, auto_encoder, test_X, test_y)
    ###################################################################################################################

    print(ae_svdd_model_params)
    print(enriched_classifier_params)
    print(enriched_classifier_params_with_vae)
    print(enriched_smote_params)
    print(enriched_smote_params_with_vae)