import pandas as pd
from pandas import DataFrame
import logging

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import inflect

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
    logger.info(X.dtypes)
    logger.info('shape of the samples feature matrix: ' + str(X.shape))
    return X


def read_processed_data(filename: str):
    processed_data = pd.read_csv(filename)
    for column in processed_data.columns:
        try:
            processed_data[column] = processed_data[column].str.replace(',', '.')
        except AttributeError:
            logger.info('str not appliable for column with key: ' + column)
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


def classify_by_abs():
    data = read_processed_data("Sprungdaten_processed/averaged_data.csv")
    train, test = train_test_split(data, test_size=0.2)

    feature_set = {('DJump_Abs_I_x LapEnd', 'DJump_Abs_I_z LapEnd'), ('DJump_SIG_I_x LapEnd', 'DJump_Abs_I_z LapEnd'),
                   ('DJump_SIG_I_x LapEnd', 'DJump_SIG_I_z LapEnd')}
    scores = dict()
    for feature in feature_set:
        X = get_samples_features(train, feature[0], feature[1])
        y = get_targets(train)
        test_actual = get_targets(test)
        clf_linear = SVC(kernel='linear')
        clf_linear.fit(X, y)
        logger.info("Train with features '" + feature[0] + ":" + feature[1] + "'")
        score = easy_prediction(clf_linear, get_samples_features(test, feature[0], feature[1]), test_actual)
        scores[feature] = score

    ranked = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    i = 0
    logger.info('Accuracy ranking from high to low:')
    p = inflect.engine()
    for key, value in ranked.items():
        i += 1
        logger.info("Feature set: " + str(key) + " has the accuracy rate of " + str(value) + ", it is the " + p.ordinal(
            i) + " best.")


def classify_by_time_splits():
    data = read_processed_data("Sprungdaten_processed/jumps_time_splits.csv.csv")
    train, test = train_test_split(data, test_size=0.2)
    X = get_samples_features(train, 'Acc_x_Fil_1', 'Gyro_z_Fil_4')
    y = get_targets(train)
    test_actual = get_targets(test)
    clf_linear = SVC(kernel='linear')
    clf_linear.fit(X, y)
    score = easy_prediction(clf_linear, get_samples_features(test, 'Acc_x_Fil_1', 'Gyro_z_Fil_4'), test_actual)
    print(score)


if __name__ == '__main__':
    classify_by_time_splits()
