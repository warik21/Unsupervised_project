from datetime import time

import optim as optim
import pandas as pd

from aeacus.src.utils.autoencoder import Autoencoder
from aeacus.src.utils.pre_proccess import prepare_data

if __name__ == '__main__':
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
    print('The process took {} seconds, of which {} went to training the autoencoder, {} went to training the first self, '
          'and {} to train the second self'.format(seconds, train_model_time, og_classifier_time, aug_classifier_time))


    plot_auto_encoder_losses(train_losses0, test_losses0)

    plot_classifier_accuracies(og_classifier_accuracies, og_eval_accuracies, "og train vs eval")
    plot_classifier_accuracies(aug_classifier_accuracies, aug_eval_accuracies, "aug train vs eval")
    plot_classifier_accuracies(og_eval_accuracies, aug_eval_accuracies, "og train vs aug train")
