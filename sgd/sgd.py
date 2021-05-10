from scipy.sparse import data
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


def read_data():
    return pd.read_csv('../Sprungdaten_processed/averaged_data.csv')


def get_target():
    return set(data['Sprungtyp'])


if __name__ == '__main__':
    train_data, test_data = train_test_split(read_data(), test_size=0.2)

    # get_features (X)
    start_column: str = 'ACC_N'
    end_column: str = 'DJump_Abs_I_z LapEnd'
    X = train_data.loc[:, start_column:end_column].to_numpy()

    # get target (y)
    y = np.array((train_data['Sprungtyp']))

    # shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    h = .02  # step size in the mesh

    clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)





