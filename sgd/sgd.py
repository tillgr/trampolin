from scipy.sparse import data
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

if __name__ == '__main__':
    train_data = pd.read_csv('../Sprungdaten_processed/avg_std_data/avg_std_data_train.csv')
    test_data = pd.read_csv('../Sprungdaten_processed/avg_std_data/avg_std_data_test.csv')

    # get_features (X)
    start_column: str = 'avg_ACC_N'
    end_column: str = 'DJump_Abs_I_z LapEnd'

    X_train = train_data.loc[:, start_column:end_column].to_numpy()
    # get target (y)
    y_train = np.array((train_data['Sprungtyp']))

    X_test = test_data.loc[:, start_column:end_column].to_numpy()
    y_test = np.array((test_data['Sprungtyp']))

    # shuffle train
    idx = np.arange(X_train.shape[0])
    np.random.seed(25)
    np.random.shuffle(idx)
    X_train_ = X_train[idx]
    y_train_ = y_train[idx]

    # shuffle test
    idx = np.arange(X_test.shape[0])
    np.random.seed(25)
    np.random.shuffle(idx)
    X_test_ = X_test[idx]
    y_test_ = y_test[idx]
    for losses in ['log', 'modified_huber', 'squared_hinge', 'perceptron']:  #Accuracy: log , l1 , 10000:  0.7558823529411764
        for penalty in ['l2', 'l1', 'elasticnet']:
            for maxi in [1000, 10000, 100000, 1000000, 10000000]:
                clf = SGDClassifier(loss=losses, penalty=penalty, alpha=0.0001,
                                    l1_ratio=0.15, fit_intercept=True, max_iter=maxi,
                                    tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                                    n_jobs=None, random_state=None, learning_rate='optimal',
                                    eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                                    n_iter_no_change=5, class_weight=None,
                                    warm_start=False, average=False).fit(X_train_, y_train_)
                y_pred = clf.predict(X_test_)
                if metrics.accuracy_score(y_test_, y_pred) > 0.74:
                    print(f"Accuracy: {losses} , {penalty} , {maxi}: ", metrics.accuracy_score(y_test_, y_pred))
