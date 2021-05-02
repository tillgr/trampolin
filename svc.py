import pandas as pd
from pandas import DataFrame
import logging

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

logging.basicConfig(filename='svc.log', format='%(asctime)s[%(name)s] - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('SVC')


def easy_classify(X: DataFrame, y: DataFrame, kernel: str):
    clf = SVC(kernel=kernel)
    clf.fit(X, y)
    return clf


def easy_prediction(classifier: SVC, testing_sample: DataFrame, test_actual: DataFrame):
    predicted = classifier.predict(testing_sample)
    score = accuracy_score(test_actual.to_numpy(), predicted)
    logger.info("prediction accuracy: " + str(score))
    print(score)


def get_samples_features(data: DataFrame, start_column: str, end_column: str):
    X: DataFrame = data.loc[:, start_column:end_column]
    X.astype(dtype='float64')
    logger.info(X.dtypes)
    logger.info('shape of the samples feature matrix: ' + str(X.shape))
    return X


def read_processed_data(filename: str):
    processed_data = pd.read_csv(filename)
    for column in processed_data.columns:
        try:
            processed_data[column] = processed_data[column].str.replace(',', '.')
        except AttributeError:
            print('str not appliable')
    return processed_data


def get_targets(data: DataFrame):
    targets = set(data['Sprungtyp'])
    logger.info('Sprungtyps are: ' + str(targets))
    logger.info('Shape of targets: ' + str(data['Sprungtyp'].shape))
    return data['Sprungtyp']


def get_jumptypes_set(data: DataFrame):
    return set(data['Sprungtyp'])


def get_jumps_by_type(data: DataFrame, type: str):
    return data[type]


if __name__ == '__main__':
    data = read_processed_data("Sprungdaten_processed/averaged_data.csv")
    train, test = train_test_split(data, test_size=0.2)
    X = get_samples_features(train, 'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_z LapEnd')
    y = get_targets(train)
    test_actual = get_targets(test)
    clf = easy_classify(X, y, 'linear')
    easy_prediction(clf, get_samples_features(test, 'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_z LapEnd'), test_actual)
