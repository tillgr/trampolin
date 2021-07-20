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

            ''' p = train_data.drop([col for col in train_data.columns if 'DJump_SIG_I_S' in col], axis=1)
                t = test_data.drop([col for col in test_data.columns if 'DJump_SIG_I_S' in col], axis=1)  '''
            # override merge phase
            train_merged = pd.read_csv(
                "../Sprungdaten_processed/without_preprocessed/percentage/" + str(
                    i) + "/vector_percentage_" + calc_type + str(
                    i) + "_train.csv")
            test_merged = pd.read_csv(
                "../Sprungdaten_processed/without_preprocessed/percentage/" + str(
                    i) + "/vector_percentage_" + calc_type + str(
                    i) + "_test.csv")

            # define columns cut
            start_column: str = '0_Acc_N_Fil'
            end_column: str = str(100 - i) + '_Gyro_z_Fil'

            if calc_type == 'mean_std_':
                start_column: str = '0_mean_Acc_N_Fil'
                end_column: str = str(100 - i) + '_std_Gyro_z_Fil'

            #p = train_merged.drop([col for col in train_merged.columns if 'DJump_ABS_I_S' in col], axis=1)
            #k = p.drop([col for col in p.columns if 'DJump_SIG_I_S' in col], axis=1)
            #o = k.drop([col for col in k.columns if 'DJump_I_ABS_S' in col], axis=1)

            #t = test_merged.drop([col for col in test_merged.columns if 'DJump_ABS_I_S' in col], axis=1)
            #l = t.drop([col for col in t.columns if 'DJump_SIG_I_S' in col], axis=1)
            #m = k.drop([col for col in l.columns if 'DJump_I_ABS_S' in col], axis=1)

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

            # TODO: gleiche anzahl an allen sprüngen?

            # params
            n_neighbors = 3
            weights = 'distance'
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
