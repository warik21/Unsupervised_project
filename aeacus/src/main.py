import time
import torch.optim as optim
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from imblearn.over_sampling import SMOTE
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from aeacus.src.utils.autoencoder import Autoencoder
from aeacus.src.utils.data_generator import generate_augmented_samples, enrich_data, DataHandler, enrich_data_smote, run_through_vae
from aeacus.src.utils.evaluation import TrainingMetrics
from aeacus.src.utils.helpers import visualize, visualize_tsne
from aeacus.src.utils.pre_proccess import prepare_data
from aeacus.src.utils.classifiers import Classifier, SVDD


if __name__ == '__main__':
    start = time.time()

    # AutoEncoder training pipeline
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path: str = r'C:\Users\eriki\Documents\school\Unsupervised_learning\big_files\creditcard.csv'
    data_frame: pd.DataFrame = pd.read_csv(data_path)
    train_X, test_X, train_y, test_y = prepare_data(data_frame)

    training_set_feature_count: int = train_X.shape[1]
    auto_encoder: Autoencoder = Autoencoder(n_features=training_set_feature_count)
    optimizer: optim.Adam = optim.Adam(auto_encoder.parameters(), lr=0.001)
    train_losses, test_losses, trained_model, train_model_time = \
        auto_encoder.train(train_X, train_y, test_X, test_y, optimizer)

    # TODO: add a run through the auto-encoder
    test_X = test_X.to(device)
    test_X = auto_encoder(test_X)


    #######################################Article suggested approach #################################################
    data_frame_good_samples: pd.DataFrame = data_frame[data_frame['Class'] == 0]
    train_ae_svdd_X, test_ae_svdd_X_without_fraud, train_ae_svdd_y, test_ae_svdd_y_without_fraud = prepare_data(data_frame_good_samples)
    train_ae_svdd_X = run_through_vae(train_ae_svdd_X, auto_encoder, device)
    train_ae_svdd_X = train_ae_svdd_X.detach().cpu()
    test_classifier = SVDD(nu=0.1, input_dim=29)
    test_classifier.train(train_ae_svdd_X)
    test_ae_svdd_X_without_fraud = run_through_vae(test_ae_svdd_X_without_fraud, auto_encoder, device)

    ae_svdd_model_params = test_classifier.eval_model(data_frame, test_ae_svdd_X_without_fraud, test_ae_svdd_y_without_fraud)
    ###################################################################################################################

    #Data enrichment overall parameters:
    n_samples = int(len(data_frame.index) / 2)

    ############################ Using latent space into classifier ####################################################
    enriched_data: DataHandler = enrich_data(data_frame, auto_encoder, n_samples)
    enriched_classifier: Classifier = Classifier()
    enriched_optimizer: optim.Adam = optim.Adam(enriched_classifier.parameters(), lr=0.001)
    enriched_data_frame: pd.DataFrame = pd.concat([enriched_data.X, pd.DataFrame(np.array(enriched_data.y).reshape(-1,1))], axis=1)
    enriched_train_X, enriched_test_X, enriched_train_y, enriched_test_y = prepare_data(enriched_data_frame)
    # enriched_train_X = run_through_vae(enriched_train_X, auto_encoder, device)
    enriched_training_metrics: TrainingMetrics = \
        enriched_classifier.train(auto_encoder.encoder, enriched_train_X, enriched_train_y, test_X, test_y, enriched_optimizer)

    # enriched_test_X = run_through_vae(enriched_test_X, auto_encoder, device)
    enriched_classifier_params = enriched_classifier.eval_model(auto_encoder, enriched_test_X, enriched_test_y)
    ###################################################################################################################

    ################################# Using SMOTE #####################################################################
    enriched_data_smote: DataHandler = enrich_data_smote(data_frame, n_samples)
    enriched_smote_classifier = Classifier()
    enriched_optimizer: optim.Adam = optim.Adam(enriched_classifier.parameters(), lr=0.001)
    enriched_data_frame: pd.DataFrame = pd.concat([enriched_data.X, pd.DataFrame(np.array(enriched_data.y).reshape(-1,1))], axis=1)
    enriched_smote_train_X, enriched_smote_test_X, enriched_smote_train_y, enriched_smote_test_y = prepare_data(enriched_data_frame)
    # enriched_smote_train_X = run_through_vae(enriched_smote_train_X, auto_encoder, device)
    enriched_smote_training_metrics: TrainingMetrics = \
        enriched_smote_classifier.train(auto_encoder.encoder, enriched_train_X, enriched_train_y, test_X, test_y, enriched_optimizer)

    # enriched_smote_test_X = run_through_vae(enriched_smote_train_X, auto_encoder, device)
    enriched_smote_params = enriched_smote_classifier.eval_model(auto_encoder, enriched_smote_test_X, enriched_smote_test_y)
    ###################################################################################################################

    cpu_tensor_list = [tensor.cpu() for tensor in enriched_training_metrics.predictions]
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
