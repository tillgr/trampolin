import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
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

    train_merged = train.merge(train_avg)
    test_merged = test.merge(test_avg)

    # percentage 10
    #train_merged = pd.read_csv("../Sprungdaten_processed/percentage/10/vector_percentage_10_train.csv")
    #test_merged = pd.read_csv("../Sprungdaten_processed/percentage/10/vector_percentage_10_test.csv")

    start_column: str = 'Acc_x_Fil_1'
    end_column: str = 'DJump_Abs_I_z LapEnd'
    #start_column: str = 'DJump_SIG_I_x LapEnd'
    #end_column: str = '10-Gyro_z_Fil'


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



    # TODO: gleiche anzahl an allen sprüngen?

    # neighbours
    n_neighbors = 15

    for weights in ['uniform', 'distance']:
        for dist_metrics in ['manhattan', 'chebyshev', 'minkowski']:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=dist_metrics)
            if dist_metrics == 'minkowski':
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=dist_metrics, p=5)

            for i in range(0, 4):
                # Train the model using the training sets
                clf.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = clf.predict(X_test)

            # compare test and predicted targets
            print(f"PARAMETER:  weights: {weights} | metric: {dist_metrics}")
            print(f"Accuracy self: ", metrics.accuracy_score(y_test, y_pred))
            print(f"Accuracy f1 score weighted: {f1_score(y_test, y_pred, average='weighted')} ")
            mean_prec, mean_rec, mean_f, mean_youden = rc_metrics(y_test, y_pred)
            print(f"Accuracy f1 score: {str(mean_f.round(5))}")
            print(f"Accuracy youden score: {str(mean_youden.round(5))}")
            print("--------------------------------------------------------------")
    # mit avg

    # Accuracy | uniform | manhattan: 0.8176470588235294
    # Accuracy | uniform | chebyshev: 0.7794117647058824
    # Accuracy | uniform | euclidean: 0.7970588235294118
    # Accuracy | distance | manhattan: 0.8558823529411764
    # Accuracy | distance | chebyshev: 0.8382352941176471
    # Accuracy | distance | euclidean: 0.8558823529411764

    # TODO: noch verschiedene dist_metriken
    # TODO: versch. Datensätze
    # TODO: Sprünge als vektoren zusammenfassen pro feature

    # mit vectorisiert und percentage

    # Accuracy | uniform | manhattan: 0.6735294117647059
    # Accuracy | uniform | chebyshev: 0.7705882352941177
    # Accuracy | uniform | euclidean: 0.7058823529411765
    # Accuracy | distance | manhattan: 0.6911764705882353
    # Accuracy | distance | chebyshev: 0.8029411764705883
    # Accuracy | distance | euclidean: 0.7323529411764705

    # mit jumos time splits

    # Accuracy | uniform | manhattan: 0.6441176470588236
    # Accuracy | uniform | chebyshev: 0.6411764705882353
    # Accuracy | uniform | euclidean: 0.6264705882352941
    # Accuracy | distance | manhattan: 0.6852941176470588
    # Accuracy | distance | chebyshev: 0.6617647058823529
    # Accuracy | distance | euclidean: 0.6441176470588236

    # splits + avg

    # Accuracy | uniform | manhattan: 0.8558823529411764
    # Accuracy | uniform | chebyshev: 0.8235294117647058
    # Accuracy | uniform | minkowski: 0.8264705882352941
    # Accuracy | distance | manhattan: 0.9058823529411765
    # Accuracy | distance | chebyshev: 0.8823529411764706
    # Accuracy | distance | minkowski: 0.8852941176470588