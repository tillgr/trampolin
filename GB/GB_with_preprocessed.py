import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from random_classifier import metrics as rc_metrics

if __name__ == '__main__':
    for i in [1, 2, 5, 10, 20, 25]:
        for calc_type in ['', 'mean_', 'mean_std_']:
            print('////////////////////////////////////////////')
            print(f"Folder: {i}")
            print(f"> Type: {calc_type}")
            # override merge phase
            train_merged = pd.read_csv(
                "../Sprungdaten_processed/with_preprocessed/percentage/" + str(i) + "/vector_percentage_" + calc_type + str(
                    i) + "_train.csv")
            test_merged = pd.read_csv(
                "../Sprungdaten_processed/with_preprocessed/percentage/" + str(i) + "/vector_percentage_" + calc_type + str(
                    i) + "_test.csv")

            # define columns cut
            start_column: str = 'DJump_SIG_I_x LapEnd'
            end_column: str = str(100 - i) + '_Gyro_z_Fil'

            if calc_type == 'mean_std_':
                end_column: str = str(100 - i) + '_std_Gyro_z_Fil'

            # get X_train
            X_train = train_merged.loc[:, start_column:end_column].to_numpy()

            # get X_test
            X_test = test_merged.loc[:, start_column:end_column].to_numpy()

            # create labelEncoder
            le = preprocessing.LabelEncoder()
            # Convert string labels into numbers

            # get y_train
            y_train = le.fit_transform(train_merged['Sprungtyp'])
            # get y_test
            y_test = le.fit_transform(test_merged['Sprungtyp'])

            # we create an instance of Neighbours Classifier and fit the data.
            for estimator in [100]:
                for depth in [1, 2]:
                    for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
                        clf = GradientBoostingClassifier(n_estimators=estimator, learning_rate=learning_rate,
                        max_depth=depth, random_state=2)

                        print('///////////////////////////////////////////////////////////////')
                        print(f"estimators: {estimator}")
                        print(f"learning_rate: {learning_rate}")
                        print(f"depth: {depth}")

                        # for i in range(0, 4):
                            # Train the model using the training sets
                        clf.fit(X_train, y_train)

                        # Predict the response for test dataset
                        y_pred = clf.predict(X_test)

                        # compare test and predicted targets
                        #print(f"PARAMETER:  weights: {weights} | metric: {dist_metrics}")
                        print(f"Accuracy self: ", metrics.accuracy_score(y_test, y_pred))
                        print(f"Accuracy f1 score weighted: {f1_score(y_test, y_pred, average='weighted')} ")
                        mean_prec, mean_rec, mean_f, mean_youden = rc_metrics(y_test, y_pred)
                        print(f"Accuracy f1 score: {str(mean_f.round(5))}")
                        print(f"Accuracy youden score: {str(mean_youden.round(5))}")
                        print("--------------------------------------------------------------")