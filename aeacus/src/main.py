import time
import torch.optim as optim
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from imblearn.over_sampling import SMOTE
from typing import Tuple

from aeacus.src.utils.autoencoder import Autoencoder
from aeacus.src.utils.data_generator import generate_augmented_samples, enrich_data, DataHandler, enrich_data_smote
from aeacus.src.utils.evaluation import TrainingMetrics
from aeacus.src.utils.helpers import visualize, visualize_tsne
from aeacus.src.utils.pre_proccess import prepare_data
from aeacus.src.utils.classifiers import Classifier, SVDD


if __name__ == '__main__':
    start = time.time()

    # AutoEncoder training pipeline
    data_path: str = r'C:\Users\eriki\Documents\school\Unsupervised_learning\big_files\creditcard.csv'
    data_frame: pd.DataFrame = pd.read_csv(data_path)
    train_X, test_X, train_y, test_y = prepare_data(data_frame)

    training_set_feature_count: int = train_X.shape[1]
    auto_encoder: Autoencoder = Autoencoder(n_features=training_set_feature_count)

    optimizer: optim.Adam = optim.Adam(auto_encoder.parameters(), lr=0.001)
    train_losses, test_losses, trained_model, train_model_time = \
        auto_encoder.train(train_X, train_y, test_X, test_y, optimizer)

    # Generating syntactic data
    n_samples = int(len(data_frame.index)/2)
    enriched_data: DataHandler = enrich_data(data_frame, auto_encoder, n_samples)

    ######## trying to enrich data with SMOTE, as used in the paper
    smote_enriched_data: DataHandler = enrich_data_smote(data_frame, n_samples)
    # TODO : Check results with SMOTE re-sampler vs our data enriching. Option - use another re-sampler
    # train_X_smote, train_y_smote = sm.fit_resample(train_X, train_y)

    encoder = trained_model.encoder

    # TODO : Train the SVDD only on the good transactions instead of both
    test_classifier = SVDD(nu=0.1, input_dim=29)
    test_classifier.train(train_X)

    enriched_classifier: Classifier = Classifier()
    enriched_optimizer: optim.Adam = optim.Adam(enriched_classifier.parameters(), lr=0.001)
    enriched_data_frame: pd.DataFrame = pd.concat([enriched_data.X, pd.DataFrame(np.array(enriched_data.y).reshape(-1,1))], axis=1)
    enriched_train_X, enriched_test_X, enriched_train_y, enriched_test_y = prepare_data(enriched_data_frame)
    enriched_training_metrics: TrainingMetrics = \
        enriched_classifier.train(encoder, enriched_train_X, enriched_train_y, test_X, test_y, enriched_optimizer)


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
