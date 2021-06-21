import pandas as pd
import shap
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from random_classifier import metrics as rc_metrics
import shap
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # read individual data sets
    train = pd.read_csv('../Sprungdaten_processed/jumps_time_splits/jumps_time_splits_train_51.csv')
    test = pd.read_csv('../Sprungdaten_processed/jumps_time_splits/jumps_time_splits_test_51.csv')

    train_2 = pd.read_csv("../Sprungdaten_processed/averaged_data/averaged_data_train.csv")
    test_2 = pd.read_csv("../Sprungdaten_processed/averaged_data/averaged_data_test.csv")


    # merge sets
    train_merged = train.merge(train_2)
    test_merged = test.merge(test_2)


    # override merge phase
    # train_merged = pd.read_csv("../Sprungdaten_processed/percentage/10/vector_percentage_10_train.csv")
    # test_merged = pd.read_csv("../Sprungdaten_processed/percentage/10/vector_percentage_10_test.csv")

    # define columns cut
    start_column: str = 'Acc_x_Fil_1'
    end_column: str = 'DJump_Abs_I_z LapEnd'

    # get X_train
    '''
    X_train = train_merged.loc[:, start_column:end_column].drop([
        '0-ACC_N_ROT_filtered',
        '10-ACC_N_ROT_filtered',
        '20-ACC_N_ROT_filtered',
        '30-ACC_N_ROT_filtered',
        '40-ACC_N_ROT_filtered',
        '50-ACC_N_ROT_filtered',
        '60-ACC_N_ROT_filtered',
        '70-ACC_N_ROT_filtered',
        '80-ACC_N_ROT_filtered',
        '90-ACC_N_ROT_filtered'
                                                                 ], axis=1).to_numpy()
    '''
    X_train = train_merged.loc[:, start_column:end_column].to_numpy()

    # get X_test
    '''
    X_test = test_merged.loc[:, start_column:end_column].drop([
        '0-ACC_N_ROT_filtered',
        '10-ACC_N_ROT_filtered',
        '20-ACC_N_ROT_filtered',
        '30-ACC_N_ROT_filtered',
        '40-ACC_N_ROT_filtered',
        '50-ACC_N_ROT_filtered',
        '60-ACC_N_ROT_filtered',
        '70-ACC_N_ROT_filtered',
        '80-ACC_N_ROT_filtered',
        '90-ACC_N_ROT_filtered'
                                                                 ], axis=1).to_numpy()
    '''
    X_test = test_merged.loc[:, start_column:end_column].to_numpy()


    # create labelEncoder
    le = preprocessing.LabelEncoder()
    # Convert string labels into numbers

    # get y_train
    y_train = le.fit_transform(train_merged['Sprungtyp'])
    # get y_test
    y_test = le.fit_transform(test_merged['Sprungtyp'])


    # parameters
    n_neighbors = 15
    weights = 'distance'
    dist_metric = 'minkowski'   # minkowski has better youden than manhattan and nearly the same accuracy (0.005
    # difference)
    dist_metric = 'manhattan'


    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=dist_metric, p=5)

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # compare test and predicted targets
    print(f"Accuracy: ", metrics.accuracy_score(y_test, y_pred).__round__(4))
    mean_prec, mean_rec, mean_f, mean_youden = rc_metrics(y_test, y_pred)
    print(f"Accuracy youden score: {str(mean_youden.round(4))}")
    print(f"Accuracy f1 score: {str(mean_f.round(4))}")
    print("--------------------------------------------------------------")

