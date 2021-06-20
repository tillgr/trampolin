import pandas as pd
from pandas import DataFrame
import logging
# import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from os import listdir, walk
from os.path import isfile, join
from random_classifier import metrics

logging.basicConfig(filename='svc_gnb.log', format='%(asctime)s[%(name)s] - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('COMMON')
gnb_logger = logging.getLogger("GNB")
svc_logger = logging.getLogger("SVC")


def easy_classify(X: DataFrame, y: DataFrame, kernel: str):
    clf = SVC(kernel=kernel)
    clf.fit(X, y)
    return clf


def prediction_and_evaludate(classifier: SVC, testing_sample: DataFrame, test_actual: DataFrame):
    predicted = classifier.predict(testing_sample)
    score = accuracy_score(test_actual.to_numpy(), predicted)
    logger.info("prediction accuracy: " + str(score.round(4)))
    f1_average = 'weighted'
    f1 = f1_score(test_actual.to_numpy(), predicted, average=f1_average)
    logger.info("F1 score with param " + f1_average + " :" + str(f1.round(4)))
    mean_prec, mean_rec, mean_f, mean_youden = metrics(test_actual.to_numpy(), predicted)
    logger.info("Random classifier: Youden score: " + str(mean_youden.round(4)))
    logger.info("Random classifier: F1 score: " + str(mean_f.round(4)))


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
    train = read_processed_data(datasets[0] + "_train.csv")
    test = read_processed_data(datasets[0] + "_test.csv")
    while next < len(datasets):
        next_train = read_processed_data(datasets[next] + "_train.csv")
        next_test = read_processed_data(datasets[next] + "_test.csv")
        train.set_index('SprungID')
        test.set_index('SprungID')
        train = train.merge(next_train)
        test = test.merge(next_test)
        next += 1
    return train, test


def classify(datasets: list, feature_start: str, feature_end: str, drops_keywords: list, reverse_drop: bool):
    train, test = get_train_test_data(datasets)

    # any(substring in string for substring in substring_list)
    drops = [col for col in train.columns if any(keyword in col for keyword in drops_keywords)]
    if reverse_drop:
        drops = [x for x in train.columns.tolist()[2:-1] if x not in drops]

    train = train.drop(columns=drops)
    test = test.drop(columns=drops)

    if feature_start == "":
        feature_start = train.columns.tolist()[2]
    if feature_end == "":
        feature_end = train.columns.tolist()[-1]

    svc_logger.info("Classify with data set: " + str(datasets) +
                ". Feature start at column: " + feature_start + ", feature end: " + feature_end +
                ". Drops :" + str(drops_keywords))

    X = get_samples_features(train, feature_start, feature_end)
    y = get_targets(train)
    test_actual = get_targets(test)
    clf_linear = SVC(kernel='linear')
    clf_linear.fit(X, y)
    prediction_and_evaludate(clf_linear, get_samples_features(test, feature_start, feature_end), test_actual)


def gnb_classify(datasets: list, feature_start: str, feature_end: str, drops_keywords: list, reverse_drop: bool):
    train, test = get_train_test_data(datasets)
    # any(substring in string for substring in substring_list)
    drops = [col for col in train.columns if any(keyword in col for keyword in drops_keywords)]
    if reverse_drop:
        drops = [x for x in train.columns.tolist()[2:-1] if x not in drops]

    train = train.drop(columns=drops)
    test = test.drop(columns=drops)

    if feature_start == "":
        feature_start = train.columns.tolist()[2]
    if feature_end == "":
        feature_end = train.columns.tolist()[-1]

    gnb_logger.info("Classify with data set: " + str(datasets) +
                ". Feature start at column: " + feature_start + ", feature end: " + feature_end +
                ". Drops :" + str(drops_keywords))

    gnb = GaussianNB()
    X = get_samples_features(train, feature_start, feature_end)
    y = get_targets(train)
    test_actual = get_targets(test)
    gnb.fit(X, y)
    prediction_and_evaludate(gnb, get_samples_features(test, feature_start, feature_end), test_actual)


def explain_model(datasets: list, feature_start: str, feature_end: str, drops_keywords: list, reverse_drop: bool):
    train, test = get_train_test_data(datasets)

    drops = [col for col in train.columns if any(keyword in col for keyword in drops_keywords)]
    if reverse_drop:
        drops = [x for x in train.columns.tolist()[2:-1] if x not in drops]

    train = train.drop(columns=drops)
    test = test.drop(columns=drops)

    if feature_start == "":
        feature_start = train.columns.tolist()[2]
    if feature_end == "":
        feature_end = train.columns.tolist()[-1]

    logger.info("Classify with data set: " + str(datasets) +
                ". Feature start at column: " + feature_start + ", feature end: " + feature_end +
                ". Drops :" + str(drops))

    X = get_samples_features(train, feature_start, feature_end)
    y = get_targets(train)
    y_test = get_targets(test)
    clf_linear = SVC(kernel='linear')
    clf_linear.fit(X, y)
    X_test = get_samples_features(test, feature_start, feature_end)

    y_pred = clf_linear.predict(X_test)
    # cm = confusion_matrix(y_test, y_pred)
    # title = 'Confusion matrix for SVC classifier'
    # disp = plot_confusion_matrix(clf_linear,
    #                              X_test,
    #                              y_test,
    #                              display_labels=set(y),
    #                              cmap=plt.cm.Blues,
    #                              normalize=None,
    #                              )
    # plt.show()
    '''
    index = y_test[y_test == 'Salto A'].index[0]
    explainer = shap.KernelExplainer(clf_linear.decision_function, X.sample(n=50), link='identity')
    df = X_test.iloc[index].to_frame().transpose()
    shap_values = explainer.shap_values(X_test.iloc[index].to_frame().transposei())
    shap.summary_plot(shap_values[0], X_test.iloc[index].to_frame().transpose())
    '''


def collect_all_data_sets(folder: str, data_sets: set):
    dirs = listdir(folder)
    for d in dirs:
        path = join(folder, d)
        if not isfile(path):
            collect_all_data_sets(path, data_sets)
        else:
            if not d.endswith("_train.csv") and not d.endswith("_test.csv") and "_AJ_" not in d:
                if "percentage" in path:
                    if d.startswith("vector_"):
                        path = path.replace(".csv", "")
                        data_sets.add(path)
                else:
                    path = path.replace(".csv", "")
                    data_sets.add(path)


def run_svc_auto(folder: str, drops: list):
    data_sets = set()
    collect_all_data_sets(folder, data_sets)
    for ds in data_sets:
        classify([ds], "", "", drops, True)


def run_gnb_auto(folder: str, drops: list):
    data_sets = set()
    collect_all_data_sets(folder, data_sets)
    for ds in data_sets:
        gnb_classify([ds], "", "", drops, False)


if __name__ == '__main__':
    folder = "Sprungdaten_processed/without_preprocessed/"
    drops_raw = []
    '''
    (0) all preprocessed data 
(1) first 9 columns (without S in name) 
(2) starts with DJump_SIG_I_S - generally worsen
(3) starts with DJump_ABS_I_S - generally worsen
(4) starts with DJump_I_ABS_S
5 all other than ["DJump_I_ABS_S", "DJump_ABS_I_S"]
6 all other than ["DJump_I_ABS_S", "DJump_SIG_I_S"]
7 all other than ["DJump_ABS_I_S", "DJump_SIG_I_S"]
'''
    logger.info("GNB Rerun")
    #run_svc_auto(folder, drops_raw)
    run_gnb_auto(folder, drops_raw)
