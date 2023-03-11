import time
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Tuple

from aeacus.src.utils.autoencoder import Autoencoder
from aeacus.src.utils.data_generator import generate_augmented_samples, enrich_data, DataHandler
from aeacus.src.utils.evaluation import TrainingMetrics
from aeacus.src.utils.helpers import visualize, visualize_tsne
from aeacus.src.utils.pre_proccess import prepare_data
from aeacus.src.utils.classifiers import Classifier


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
    n_samples = 27000
    enriched_data: DataHandler = enrich_data(data_frame, auto_encoder, n_samples)

    encoder = trained_model.encoder

    # Naive classifier training pipeline
    original_classifier: Classifier = Classifier()
    original_optimizer: optim.Adam = optim.Adam(original_classifier.parameters(), lr=0.001)

    original_training_metrics: TrainingMetrics = \
        original_classifier.train(encoder, train_X, train_y, test_X, test_y, original_optimizer)


    print('hello world')
    # og_classifier_losses, og_classifier_accuracies, og_eval_losses, og_eval_accuracies, og_classifier_time = \

    # aug_data_path = r'C:\Users\eriki\Documents\school\Unsupervised_learning\big_files\rand.csv'
    # aug_data_frame = pd.read_csv(aug_data_path)
    # aug_train_X, aug_test_X, aug_train_y, aug_test_y = prepare_data(aug_data_frame)
    # aug_classifier = Classifier()
    # aug_optimizer = optim.Adam(aug_classifier.parameters(), lr=0.001)
    # aug_classifier_losses, aug_classifier_accuracies, aug_eval_losses, aug_eval_accuracies, aug_classifier_time = \
    #     aug_classifier.train(encoder, aug_train_X, aug_train_y, test_X, test_y, aug_optimizer)
    #
    # end = time.time()
    # seconds = end - start

    # print('The process took {} seconds, of which {} went to training the autoencoder, {} went to training the first self, '
    #       'and {} to train the second self'.format(seconds, train_model_time, og_classifier_time, aug_classifier_time))


    # plot_auto_encoder_losses(train_losses0, test_losses0)
#
    # plot_classifier_accuracies(og_classifier_accuracies, og_eval_accuracies, "og train vs eval")
    # plot_classifier_accuracies(aug_classifier_accuracies, aug_eval_accuracies, "aug train vs eval")
    # plot_classifier_accuracies(og_eval_accuracies, aug_eval_accuracies, "og train vs aug train")
