import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing

if __name__ == '__main__':
    #read data
    data = pd.read_csv('../Sprungdaten_processed/averaged_data.csv')

    #get targets
    # creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    targets = le.fit_transform(data['Sprungtyp'])

    # Split dataset into training set and test set
    start_column: str = 'ACC_N'
    end_column: str = 'DJump_Abs_I_z LapEnd'
    X_train, X_test, y_train, y_test = train_test_split(
        data.loc[:, start_column:end_column].to_numpy(),
        targets,
        test_size=0.3)

    #neighbours
    n_neighbors = 15

    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

        # Train the model using the training sets
        clf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))