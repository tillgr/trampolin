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

    for i in [10]:  # [1, 2, 5, 10, 20, 25]
        for calc_type in ['mean_']:  # ['', 'mean_', 'mean_std_']
            print('////////////////////////////////////////////')
            print(f"Folder: {i}")
            print(f"> Type: {calc_type}")
            # override merge phase
            train_merged = pd.read_csv(
                "../Sprungdaten_processed/with_preprocessed/percentage/" + str(
                    i) + "/vector_percentage_" + calc_type + str(
                    i) + "_train.csv")
            test_merged = pd.read_csv(
                "../Sprungdaten_processed/with_preprocessed/percentage/" + str(
                    i) + "/vector_percentage_" + calc_type + str(
                    i) + "_test.csv")
            # data = pd.read_csv(
            #         "../Sprungdaten_processed/with_preprocessed/percentage/" + str(
            #             i) + "/vector_percentage_" + calc_type + str(
            #             i) + ".csv")


            # # define columns cut
            # start_column: str = 'DJump_SIG_I_x LapEnd'
            # end_column: str = str(100 - i) + '_Gyro_z_Fil'
            #
            # if calc_type == 'mean_std_':
            #     end_column: str = str(100 - i) + '_std_Gyro_z_Fil'
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

            # we create an instance of Neighbours Classifier and fit the data.
            # for estimator in [100]:
            #     for depth in [1, 2, 3, 4, 5]:
            #         for learning_rate in [0.1, 0.01]:
            #             clf = GradientBoostingClassifier(n_estimators=estimator, learning_rate=learning_rate,
            #             max_depth=depth, random_state=2)
            #
            #             # for i in range(0, 4):
            #                 # Train the model using the training sets
            #             clf.fit(X_train, y_train)
            #
            #             # Predict the response for test dataset
            #             y_pred = clf.predict(X_test)
            #
            #             # compare test and predicted targets
            #             print(f"PARAMETER:  estimators: {estimator} | learning_rate: {learning_rate} | depth: {depth}")
            #             print(f"Accuracy self: ", metrics.accuracy_score(y_test, y_pred))
            #             #print(f"Accuracy f1 score weighted: {f1_score(y_test, y_pred, average='weighted')} ")
            #             mean_prec, mean_rec, mean_f, mean_youden = rc_metrics(y_test, y_pred)
            #             print(f"Accuracy f1 score: {str(mean_f.round(5))}")
            #             print(f"Accuracy youden score: {str(mean_youden.round(5))}")
            #             print("--------------------------------------------------------------")


            # for i in range(0, 4):
            # Train the model using the training sets
            # clf.fit(X_train, y_train)

            # Predict the response for test dataset
            # y_pred = clf.predict(X_test)

            model = GradientBoostingClassifier()

            param_dist = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200],
                          'max_depth': [3, 4, 5, 6, 7]}
            run=20

            grid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, verbose=1, n_iter=run,  n_jobs=7, cv = 2)
            grid_result = grid.fit(X_test, y_test)
            print(grid_result.best_params_)


            # compare test and predicted targets
            # print(f"PARAMETER:  estimators: {estimator} | learning_rate: {learning_rate} | depth: {depth}")
            # print(f"Accuracy self: ", metrics.accuracy_score(y_test, y_pred))
            # mean_prec, mean_rec, mean_f, mean_youden = rc_metrics(y_test, y_pred)
            # print(f"Accuracy f1 score: {str(mean_f.round(5))}")
            # print(f"Accuracy youden score: {str(mean_youden.round(5))}")
            # print("--------------------------------------------------------------")
