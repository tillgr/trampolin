import pandas as pd
from pandas import DataFrame
import logging

from sklearn.metrics import accuracy_score
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
    return score


def get_samples_features(data: DataFrame, start_column: str, end_column: str):
    X: DataFrame = data.loc[:, start_column:end_column]
    X.astype(dtype='float64')
    # logger.info(X.dtypes)
    # logger.info('shape of the samples feature matrix: ' + str(X.shape))
    return X


def read_processed_data(filename: str):
    processed_data = pd.read_csv(filename)
    logger.info("read data set:" + filename)
    for column in processed_data.columns:
        try:
            processed_data[column] = processed_data[column].str.replace(',', '.')
        except AttributeError:
            pass
    return processed_data


def get_targets(data: DataFrame):
    targets = set(data['Sprungtyp'])
    return data['Sprungtyp']


def get_jumptypes_set(data: DataFrame):
    return set(data['Sprungtyp'])


def get_jumps_by_type(data: DataFrame, type: str):
    return data[type]


def get_train_test_data(datasets: list):
    next = 1
    train = read_processed_data("Sprungdaten_processed/" + datasets[0] + "_train.csv")
    test = read_processed_data("Sprungdaten_processed/" + datasets[0] + "_test.csv")
    while next < len(datasets):
        next_train = read_processed_data("Sprungdaten_processed/" + datasets[next] + "_train.csv")
        next_test = read_processed_data("Sprungdaten_processed/" + datasets[next] + "_test.csv")
        train.set_index('SprungID')
        test.set_index('SprungID')
        train = train.merge(next_train)
        test = test.merge(next_test)
        next += 1
    return train, test


def classify(datasets: list, feature_start: str, feature_end: str, drops: list):
    train, test = get_train_test_data(datasets)
    logger.info("Classify with data set: " + str(datasets) +
                ". Feature start at column: " + feature_start + ", feature end: " + feature_end +
                ". Drops :" + str(drops))
    train = train.drop(columns=drops)
    test = test.drop(columns=drops)
    X = get_samples_features(train, feature_start, feature_end)
    y = get_targets(train)
    test_actual = get_targets(test)
    clf_linear = SVC(kernel='linear')
    clf_linear.fit(X, y)
    easy_prediction(clf_linear, get_samples_features(test, feature_start, feature_end), test_actual)


if __name__ == '__main__':
    logger.info("------------start of new run------------")
    classify(['percentage/2/vector_percentage_2'], '0-ACC_N', '98-Gyro_z_Fil', [])
    classify(['percentage/2/vector_percentage_2'], 'DJump_SIG_I_x LapEnd', '98-Gyro_z_Fil', [])
    classify(['percentage/2/vector_percentage_2', 'avg_std_data/avg_std_data'], 'DJump_SIG_I_x LapEnd', 'std_Gyro_z_Fil', [])
    i = 0
    drops = []
    while i < 100:
        drops.append(str(i) + "-ACC_N")
        drops.append(str(i) + "-ACC_N_ROT_filtered")
        i += 2
    classify(['percentage/2/vector_percentage_2', 'avg_std_data/avg_std_data'], 'DJump_SIG_I_x LapEnd', 'std_Gyro_z_Fil', drops)