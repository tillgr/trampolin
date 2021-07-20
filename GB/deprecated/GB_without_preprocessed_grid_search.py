import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn import neighbors
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, classification_report
from random_classifier import metrics as rc_metrics
from time import time
from sklearn.ensemble import GradientBoostingClassifier as GBC

if __name__ == '__main__':

    for i in [1, 2, 5, 10, 20, 25]:  # [1, 2, 5, 10, 20, 25]
        for calc_type in ['', 'mean_', 'mean_std_']:  # ['', 'mean_', 'mean_std_']
            print('////////////////////////////////////////////')
            print(f"Folder: {i}")
            print(f"> Type: {calc_type}")
            # override merge phase
            train_merged = pd.read_csv(
                "../Sprungdaten_processed/without_preprocessed/percentage/" + str(
                    i) + "/vector_percentage_" + calc_type + str(
                    i) + "_train.csv")
            test_merged = pd.read_csv(
                "../Sprungdaten_processed/without_preprocessed/percentage/" + str(
                    i) + "/vector_percentage_" + calc_type + str(
                    i) + "_test.csv")


            # # define columns cut
            # start_column: str = 'DJump_SIG_I_x LapEnd'
            # end_column: str = str(100 - i) + '_Gyro_z_Fil'
            #
            #
            # # get X_train
            # X_train = train_merged.loc[:, start_column:end_column]
            #
            # # get X_test
            # X_test = test_merged.loc[:, start_column:end_column]
            X_train = train_merged.drop(['SprungID', 'Sprungtyp'], axis=1)
            X_test = test_merged.drop(['SprungID', 'Sprungtyp'], axis=1)

            # create labelEncoder
            le = preprocessing.LabelEncoder()
            # Convert string labels into numbers

            # get y_train
            y_train = le.fit_transform(train_merged['Sprungtyp'])
            # get y_test
            y_test = le.fit_transform(test_merged['Sprungtyp'])


            model = GradientBoostingClassifier()

            param_dist = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200],
                          'max_depth': [3, 4, 5, 6, 7]}
            n_iter=15

            grid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, verbose=1, n_iter=n_iter,  n_jobs=7, cv = 2)
            grid_result = grid.fit(X_test, y_test)
            print(f"Params: {grid_result.best_params_}")
            print(f"Score: {grid_result.best_score_}")
