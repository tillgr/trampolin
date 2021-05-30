from scipy.sparse import data
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from random_classifier import metrics


def test_1():
    train_data = pd.read_csv('../Sprungdaten_processed/jumps_time_splits/jumps_time_splits_train_51.csv')
    test_data = pd.read_csv('../Sprungdaten_processed/jumps_time_splits/jumps_time_splits_test_51.csv')

    # get_features (X)
    start_column: str = 'Acc_x_Fil_1'
    end_column: str = 'Gyro_z_Fil_50'

    '''start_column: str = 'ACC_N'
    end_column: str = 'Gyro_z_Fil' '''

    X_train = train_data.loc[:, start_column:end_column].to_numpy()
    # get target (y)
    y_train = np.array((train_data['Sprungtyp']))

    X_test = test_data.loc[:, start_column:end_column].to_numpy()
    y_test = np.array((test_data['Sprungtyp']))

def test2():

    train_data = pd.read_csv('../Sprungdaten_processed/jumps_time_splits/jumps_time_splits_train_21.csv')
    test_data = pd.read_csv('../Sprungdaten_processed/jumps_time_splits/jumps_time_splits_test_21.csv')

    # get_features (X)
    start_column: str = 'Acc_x_Fil_1'
    end_column: str = 'DJump_Abs_I_z LapEnd'

    X_train = train_data.loc[:, start_column:end_column].to_numpy()
    y_train = np.array((train_data['Sprungtyp']))

    X_test = test_data.loc[:, start_column:end_column].to_numpy()
    y_test = np.array((test_data['Sprungtyp']))


if __name__ == '__main__':

    '''train = pd.read_csv("../Sprungdaten_processed/jumps_time_splits/jumps_time_splits_train_101.csv")
    test = pd.read_csv("../Sprungdaten_processed/jumps_time_splits/jumps_time_splits_test_101.csv")
    train_avg = pd.read_csv("../Sprungdaten_processed/averaged_data/averaged_data_train.csv")
    test_avg = pd.read_csv("../Sprungdaten_processed/averaged_data/averaged_data_test.csv")

    train_merged = train.merge(train_avg)
    test_merged = test.merge(test_avg)

    start_column: str = 'Acc_x_Fil_1'
    end_column: str = 'DJump_Abs_I_z LapEnd'

    X_train = train_merged.loc[:, start_column:end_column].to_numpy()
    y_train = np.array((train_merged['Sprungtyp']))

    X_test = test_merged.loc[:, start_column:end_column].to_numpy()
    y_test = np.array((test_merged['Sprungtyp']))'''

    train_data = pd.read_csv('../Sprungdaten_processed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv('../Sprungdaten_processed/percentage/10/vector_percentage_mean_std_10_test.csv')

    # get_features (X)
    start_column: str = 'DJump_SIG_I_x LapEnd'
    end_column: str = '90-std_Gyro_z_Fil'

    X_train = train_data.loc[:, start_column:end_column].to_numpy()
    y_train = np.array((train_data['Sprungtyp']))

    X_test = test_data.loc[:, start_column:end_column].to_numpy()
    y_test = np.array((test_data['Sprungtyp']))

    for losses in ['log', 'modified_huber', 'squared_hinge', 'perceptron']:
        for penalty in ['l2', 'l1', 'elasticnet']:
            for maxi in [10000]:
                clf = SGDClassifier(loss=losses, penalty=penalty, alpha=0.0001,
                                    l1_ratio=0.15, fit_intercept=True, max_iter=maxi,
                                    tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                                    n_jobs=None, random_state=10, learning_rate='optimal',
                                    eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                                    n_iter_no_change=5, class_weight=None,
                                    warm_start=False, average=False).fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                if accuracy_score(y_test, y_pred) > 0.89:
                    print(f"Accuracy score: {losses} , {penalty} , {maxi}: ", accuracy_score(y_test, y_pred))
                    print(f"Accuracy f1 score weighted: {f1_score(y_test, y_pred, average='weighted')} ")
                    mean_prec, mean_rec, mean_f, mean_youden = metrics(y_test, y_pred)
                    print(f"Accuracy f1 score: {str(mean_f.round(5))}")
                    print(f"Accuracy youden score: {str(mean_youden.round(5))}")
