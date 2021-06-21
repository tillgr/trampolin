import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from random_classifier import metrics as rc_metrics

if __name__ == '__main__':
    # read data
    '''
    train = pd.read_csv('../Sprungdaten_processed/percentage/5/vector_percentage_mean_5_train.csv')
    test = pd.read_csv('../Sprungdaten_processed/percentage/5/vector_percentage_mean_5_test.csv')
    '''

    # jumps_time_splits
    test = pd.read_csv('../Sprungdaten_processed/jumps_time_splits/jumps_time_splits_test_51.csv')
    train = pd.read_csv('../Sprungdaten_processed/jumps_time_splits/jumps_time_splits_train_51.csv')

    train_avg = pd.read_csv("../Sprungdaten_processed/avg_std_data/avg_std_data_train.csv")
    test_avg = pd.read_csv("../Sprungdaten_processed/avg_std_data/avg_std_data_test.csv")

    # define data columns
    '''
    start_column: str = 'DJump_SIG_I_x LapEnd'
    end_column: str = '95-Gyro_z_Fil'
    '''
    '''
    # get X_train
    X_train = train.loc[:, start_column:end_column].to_numpy()
    # get X_test
    X_test = test.loc[:, start_column:end_column].to_numpy()
    '''

    #train_merged = train.merge(train_avg)
    #test_merged = test.merge(test_avg)

    # percentage 10
    train_merged = pd.read_csv("../Sprungdaten_processed/percentage/10/vector_percentage_10_train.csv")
    test_merged = pd.read_csv("../Sprungdaten_processed/percentage/10/vector_percentage_10_test.csv")

    # start_column: str = 'Acc_x_Fil_1'
    # end_column: str = 'DJump_Abs_I_z LapEnd'
    start_column: str = 'DJump_SIG_I_x LapEnd'
    end_column: str = '90-Gyro_z_Fil'


    # get X_train
    X_train = train_merged.loc[:, start_column:end_column].to_numpy()
    # get X_test
    X_test = test_merged.loc[:, start_column:end_column].to_numpy()


    # create labelEncoder
    le = preprocessing.LabelEncoder()
    # Convert string labels into numbers

    # get y_train
    y_train = le.fit_transform(train['Sprungtyp'])
    # get y_test
    y_test = le.fit_transform(test['Sprungtyp'])


    # we create an instance of Neighbours Classifier and fit the data.
    clf = GaussianNB()
    clf_pf = GaussianNB()

    for i in range(0, 4):
        # Train the model using the training sets
        clf.fit(X_train, y_train)
        clf_pf.partial_fit(X_train, y_train, np.unique(y_train))

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    y_pred_pf = clf_pf.predict(X_test)

    # compare test and predicted targets
    #print(f"PARAMETER:  weights: {weights} | metric: {dist_metrics}")
    print(f"Accuracy self: ", metrics.accuracy_score(y_test, y_pred))
    print(f"Accuracy f1 score weighted: {f1_score(y_test, y_pred, average='weighted')} ")
    mean_prec, mean_rec, mean_f, mean_youden = rc_metrics(y_test, y_pred)
    print(f"Accuracy f1 score: {str(mean_f.round(5))}")
    print(f"Accuracy youden score: {str(mean_youden.round(5))}")
    print("--------------------------------------------------------------")

    print(f"Accuracy self: ", metrics.accuracy_score(y_test, y_pred_pf))
    print(f"Accuracy f1 score weighted: {f1_score(y_test, y_pred_pf, average='weighted')} ")
    mean_prec, mean_rec, mean_f, mean_youden = rc_metrics(y_test, y_pred_pf)
    print(f"Accuracy f1 score: {str(mean_f.round(5))}")
    print(f"Accuracy youden score: {str(mean_youden.round(5))}")
    print("==============================================================")