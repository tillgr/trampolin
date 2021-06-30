import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
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
                start_column: str = 'DJump_SIG_I_x LapEnd'
                end_column: str = str(100 - i) + '_std_Gyro_z_Fil'



            # get X_train
            X_train = train_merged.loc[:, start_column:end_column]

            # get X_test
            X_test = test_merged.loc[:, start_column:end_column]

            # create labelEncoder
            le = preprocessing.LabelEncoder()
            # Convert string labels into numbers

            # get y_train
            y_train = le.fit_transform(train_merged['Sprungtyp'])
            # get y_test
            y_test = le.fit_transform(test_merged['Sprungtyp'])

            # params
            estimators = 200
            depth = 7

            clf = GradientBoostingClassifier(n_estimators=estimators, max_depth=depth)

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
