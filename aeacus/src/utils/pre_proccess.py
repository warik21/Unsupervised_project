import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def prepare_data(df, train_ratio = 0.8):
    """
    Prepares the data for training the model.

    Args:
        df: The dataframe
        train_ratio: the ratio of the training set

    Returns:
        A tuple containing the preprocessed training data and labels.
    """
    df = df.fillna(1)
    if 'Time' in df.columns:
        df = df.drop('Time', axis=1)
    x = df[df.columns[:len(df.columns) - 1]].to_numpy()
    y = df[df.columns[len(df.columns) - 1]].to_numpy()

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

    scaler_train = preprocessing.MinMaxScaler()
    scaler_train.fit(train_x)
    scaled_x_train = scaler_train.transform(train_x)

    # Convert numpy arrays to tensors
    train_x = torch.tensor(scaled_x_train, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    scaler_test = preprocessing.MinMaxScaler()
    scaler_test.fit(test_x)
    scaled_x_test = scaler_test.transform(test_x)

    test_x = torch.tensor(scaled_x_test, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    return train_x, test_x, train_y, test_y
