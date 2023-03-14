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
    train_vae


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
    ae_svdd_model_params = suggested_approach(data_frame, auto_encoder, device)
    ###################################################################################################################

    #Data enrichment overall parameters:
    n_samples = int(len(data_frame.index) / 2)

    ############################ Using latent space into classifier ####################################################
    enriched_classifier_params = latent_enrichment_approach(data_frame, auto_encoder, n_samples, test_X, test_y)
    ###################################################################################################################

    ################################# Using SMOTE #####################################################################
    enriched_smote_params = smote_enrichment_approach(data_frame, auto_encoder, test_X, test_y)
    ###################################################################################################################


    cpu_tensor_list = [tensor.cpu() for tensor in test_y]
    y_pred = np.stack([tensor.detach().numpy() for tensor in cpu_tensor_list])
    precision, recall, _ = precision_recall_curve(test_y, y_pred)
    pr_auc = auc(recall, precision)
    prauc_score = average_precision_score(test_y, y_pred, average='macro')
    plt.plot(recall, precision, label='PRAUC = %0.2f' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='lower left')
    plt.show()
    print('hello world')

    # end = time.time()
    # seconds = end - start

    # print('The process took {} seconds, of which {} went to training the autoencoder, {} went to training the first self, '
    #       'and {} to train the second self'.format(seconds, train_model_time, og_classifier_time, aug_classifier_time))


    # plot_auto_encoder_losses(train_losses0, test_losses0)
#
    # plot_classifier_accuracies(og_classifier_accuracies, og_eval_accuracies, "og train vs eval")
    # plot_classifier_accuracies(aug_classifier_accuracies, aug_eval_accuracies, "aug train vs eval")
    # plot_classifier_accuracies(og_eval_accuracies, aug_eval_accuracies, "og train vs aug train")
