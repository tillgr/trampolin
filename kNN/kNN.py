import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing

if __name__ == '__main__':
    # read data
    train = pd.read_csv('../Sprungdaten_processed/averaged_data/averaged_data_train.csv')
    test = pd.read_csv('../Sprungdaten_processed/averaged_data/averaged_data_test.csv')

    # define data columns
    start_column: str = 'ACC_N'
    end_column: str = 'DJump_Abs_I_z LapEnd'

    # get X_train
    X_train = train.loc[:, start_column:end_column].to_numpy()
    # get X_test
    X_test = test.loc[:, start_column:end_column].to_numpy()

    # create labelEncoder
    le = preprocessing.LabelEncoder()
    # Convert string labels into numbers

    # get y_train
    y_train = le.fit_transform(train['Sprungtyp'])
    # get y_test
    y_test = le.fit_transform(test['Sprungtyp'])

    # gleiche anzahl an allen sprüngen?

    # neighbours
    n_neighbors = 15

    for weights in ['uniform', 'distance']:
        for dist_metrics in ['manhattan', 'chebyshev', 'euclidean']:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, metric=dist_metrics)

            # Train the model using the training sets
            clf.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = clf.predict(X_test)

            # compare test and predicted targets
            print(f"Accuracy | {weights} | {dist_metrics}:", metrics.accuracy_score(y_test, y_pred))
