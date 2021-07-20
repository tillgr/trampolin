import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from random_classifier import metrics as rc_metrics

if __name__ == '__main__':

            '''train_merged = pd.read_csv('../Sprungdaten_processed/without_preprocessed/averaged_data/averaged_data_train.csv')
            test_merged = pd.read_csv('../Sprungdaten_processed/without_preprocessed/averaged_data/averaged_data_test.csv')
            start_column: str = 'avg_Acc_N_Fil'
            end_column: str = 'avg_Gyro_z_Fil' '''

            '''train_merged = pd.read_csv('../Sprungdaten_processed/with_preprocessed/averaged_data/averaged_data_train.csv')
            test_merged = pd.read_csv('../Sprungdaten_processed/with_preprocessed/averaged_data/averaged_data_test.csv')
            start_column: str = 'avg_Acc_N_Fil'
            end_column: str = 'avg_DJump_ABS_I_S4_z LapEnd' '''

            '''train_merged = pd.read_csv('../Sprungdaten_processed/with_preprocessed/std_data/std_data_train.csv')
            test_merged = pd.read_csv('../Sprungdaten_processed/with_preprocessed/std_data/std_data_test.csv')
            start_column: str = 'std_Acc_N_Fil'
            end_column: str = 'std_Gyro_z_Fil' '''

            '''train_merged = pd.read_csv('../Sprungdaten_processed/without_preprocessed/std_data/std_data_train.csv')
            test_merged = pd.read_csv('../Sprungdaten_processed/without_preprocessed/std_data/std_data_test.csv')
            start_column: str = 'std_Acc_N_Fil'
            end_column: str = 'std_Gyro_z_Fil' '''


            '''train_merged = pd.read_csv('../Sprungdaten_processed/with_preprocessed/avg_std_data/avg_std_data_train.csv')
            test_merged = pd.read_csv('../Sprungdaten_processed/with_preprocessed/avg_std_data/avg_std_data_test.csv')
            start_column: str = 'avg_Acc_N_Fil'
            end_column: str = 'avg_DJump_ABS_I_S4_z LapEnd' '''

            train_merged = pd.read_csv(
                '../../Sprungdaten_processed/without_preprocessed/avg_std_data/avg_std_data_train.csv')
            test_merged = pd.read_csv(
                '../../Sprungdaten_processed/without_preprocessed/avg_std_data/avg_std_data_test.csv')
            start_column: str = 'avg_Acc_N_Fil'
            end_column: str = 'std_Gyro_z_Fil'


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
