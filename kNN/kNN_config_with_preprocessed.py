import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
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
                "../Sprungdaten_processed/with_preprocessed/percentage/" + str(
                    i) + "/vector_percentage_" + calc_type + str(
                    i) + "_train.csv")
            test_merged = pd.read_csv(
                "../Sprungdaten_processed/with_preprocessed/percentage/" + str(
                    i) + "/vector_percentage_" + calc_type + str(
                    i) + "_test.csv")

            # define columns cut
            start_column: str = 'DJump_SIG_I_x LapEnd'
            end_column: str = str(100 - i) + '_Gyro_z_Fil'

            if calc_type == 'mean_std_':
                end_column: str = str(100 - i) + '_std_Gyro_z_Fil'

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

            # TODO: gleiche anzahl an allen sprüngen?

            # neighbours
            #n_neighbors = 15

            for weights in ['uniform', 'distance']:
                for dist_metrics in ['manhattan', 'chebyshev', 'minkowski']:
                    for n_neighbors in [3, 5, 7, 9, 11, 13, 15]:
                        # we create an instance of Neighbours Classifier and fit the data.
                        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=dist_metrics)
                        if dist_metrics == 'minkowski':
                            clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,
                                                                 metric=dist_metrics, p=5)

                        clf.fit(X_train, y_train)

                        # Predict the response for test dataset
                        y_pred = clf.predict(X_test)

                        # compare test and predicted targets
                        if metrics.accuracy_score(y_test, y_pred) >= 0.9:
                            print(f"PARAMETER:  weights: {weights} | metric: {dist_metrics}  | neighbours: {n_neighbors}")
                            print(f"Accuracy self: ", metrics.accuracy_score(y_test, y_pred))
                            print(f"Accuracy f1 score weighted: {f1_score(y_test, y_pred, average='weighted')} ")
                            mean_prec, mean_rec, mean_f, mean_youden = rc_metrics(y_test, y_pred)
                            print(f"Accuracy f1 score: {str(mean_f.round(5))}")
                            print(f"Accuracy youden score: {str(mean_youden.round(5))}")
                            print("--------------------------------------------------------------")
