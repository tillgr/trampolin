import pandas as pd
from pandas import DataFrame
import logging
import shap

from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

from random_classifier import metrics

logging.basicConfig(filename='svc_gnb.log', format='%(asctime)s[%(name)s] - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('SVC')
gnb_logger = logging.getLogger("GNB")


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
    prediction_and_evaludate(clf_linear, get_samples_features(test, feature_start, feature_end), test_actual)


def gnb_classify(datasets: list, feature_start: str, feature_end: str, drops: list):
    train, test = get_train_test_data(datasets)
    gnb = GaussianNB()
    X = get_samples_features(train, feature_start, feature_end)
    y = get_targets(train)
    test_actual = get_targets(test)
    gnb.fit(X, y)
    prediction_and_evaludate(gnb, get_samples_features(test, feature_start, feature_end), test_actual)


def explain_model(datasets: list, feature_start: str, feature_end: str, drops: list):
    train, test = get_train_test_data(datasets)
    logger.info("Classify with data set: " + str(datasets) +
                ". Feature start at column: " + feature_start + ", feature end: " + feature_end +
                ". Drops :" + str(drops))
    train = train.drop(columns=drops)
    test = test.drop(columns=drops)
    X = get_samples_features(train, feature_start, feature_end)
    y = get_targets(train)
    y_test = get_targets(test)
    clf_linear = SVC(kernel='linear')
    clf_linear.fit(X, y)
    X_test = get_samples_features(test, feature_start, feature_end)
    index = y_test[y_test == 'Salto A'].index[0]
    explainer = shap.KernelExplainer(clf_linear.decision_function, X.sample(n=50), link='identity')
    shap_values = explainer.shap_values(X_test.sample(n=10))
    shap.summary_plot(shap_values[0], X_test.sample(n=10))

    # shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], link="logit")

def run_svc():
    # logger.info("------------start of new run------------")
    # classify(['percentage/2/vector_percentage_2'], '0-ACC_N', '98-Gyro_z_Fil', [])
    # logger.info("------------p2------------")
    #
    # classify(['percentage/2/vector_percentage_2'], 'DJump_SIG_I_x LapEnd', '98-Gyro_z_Fil', [])
    # classify(['percentage/2/vector_percentage_2', 'avg_std_data/avg_std_data'], 'DJump_SIG_I_x LapEnd', 'std_Gyro_z_Fil', [])
    # i = 0
    # drops = []
    # while i < 100:
    #     drops.append(str(i) + "-ACC_N")
    #     drops.append(str(i) + "-ACC_N_ROT_filtered")
    #     i += 2
    # classify(['percentage/2/vector_percentage_2', 'avg_std_data/avg_std_data'], 'DJump_SIG_I_x LapEnd', 'std_Gyro_z_Fil', drops)
    # classify(['percentage/2/vector_percentage_mean_2'], '0-ACC_N', '98-Gyro_z_Fil', [])
    # logger.info("------------p2 mean------------")
    # classify(['percentage/2/vector_percentage_mean_2'], 'DJump_SIG_I_x LapEnd', '98-Gyro_z_Fil', [])
    # logger.info("------------p2 std------------")
    # classify(['percentage/2/vector_percentage_mean_std_2'], 'DJump_SIG_I_x LapEnd', '98-std_Gyro_z_Fil', [])
    # logger.info("------------p1 ------------")
    # classify(['percentage/1/vector_percentage_1'], 'DJump_SIG_I_x LapEnd', '99-Gyro_z_Fil', [])
    # logger.info("------------p1 mean------------")
    # classify(['percentage/1/vector_percentage_mean_1'], 'DJump_SIG_I_x LapEnd', '99-Gyro_z_Fil', [])
    # logger.info("------------p1 std------------")
    # classify(['percentage/1/vector_percentage_mean_std_1'], 'DJump_SIG_I_x LapEnd', '99-std_Gyro_z_Fil', [])
    # logger.info("------------p5 ------------")
    # classify(['percentage/5/vector_percentage_5'], 'DJump_SIG_I_x LapEnd', '95-Gyro_z_Fil', [])
    # logger.info("------------p5 mean------------")
    # classify(['percentage/5/vector_percentage_mean_5'], 'DJump_SIG_I_x LapEnd', '95-Gyro_z_Fil', [])
    # logger.info("------------p5 std------------")
    # classify(['percentage/5/vector_percentage_mean_std_5'], 'DJump_SIG_I_x LapEnd', '95-std_Gyro_z_Fil', [])
    # logger.info("------------p10 ------------")
    # classify(['percentage/10/vector_percentage_10'], 'DJump_SIG_I_x LapEnd', '90-Gyro_z_Fil', [])
    # logger.info("------------p10 mean------------")
    # classify(['percentage/10/vector_percentage_mean_10'], 'DJump_SIG_I_x LapEnd', '90-Gyro_z_Fil', [])
    # logger.info("------------p10 std------------")
    # classify(['percentage/10/vector_percentage_mean_std_10'], 'DJump_SIG_I_x LapEnd', '90-std_Gyro_z_Fil', [])

    # logger.info("------------p20 ------------")
    # classify(['percentage/20/vector_percentage_20'], 'DJump_SIG_I_x LapEnd', '80-Gyro_z_Fil', [])
    # logger.info("------------p20 mean------------")
    # classify(['percentage/20/vector_percentage_mean_20'], 'DJump_SIG_I_x LapEnd', '80-Gyro_z_Fil', [])
    # logger.info("------------p20 std------------")
    #     # classify(['percentage/20/vector_percentage_mean_std_20'], 'DJump_SIG_I_x LapEnd', '80-std_Gyro_z_Fil', [])
    # logger.info("------------p20 std-----only processed-------")
    # classify(['percentage/20/vector_percentage_mean_std_20'], 'DJump_SIG_I_x LapEnd', 'DJump_Abs_I_z LapEnd', [])

    i = 0
    drops = []
    while i < 100:
        drops.append(str(i) + "-mean_ACC_N_ROT_filtered")
        drops.append(str(i) + "-std_ACC_N_ROT_filtered")
        i += 20
    logger.info("------------p20 std-----without processed, without n rot filtered-------")
    explain_model(['percentage/20/vector_percentage_mean_std_20'], '0-mean_ACC_N', '80-std_Gyro_z_Fil', drops)

    # logger.info("------------p25 ------------")
    # classify(['percentage/25/vector_percentage_25'], 'DJump_SIG_I_x LapEnd', '75-Gyro_z_Fil', [])
    # logger.info("------------p25 mean------------")
    # classify(['percentage/25/vector_percentage_mean_25'], 'DJump_SIG_I_x LapEnd', '75-Gyro_z_Fil', [])
    # logger.info("------------p25 std------------")
    # classify(['percentage/25/vector_percentage_mean_std_25'], 'DJump_SIG_I_x LapEnd', '75-std_Gyro_z_Fil', [])

    '''
    logger.info("------------avg ------------")
    classify(['averaged_data/averaged_data'], 'ACC_N', 'DJump_Abs_I_z LapEnd', [])
    logger.info("------------avg std------------")
    classify(['avg_std_data/avg_std_data'], 'avg_ACC_N', 'DJump_Abs_I_z LapEnd', [])
    logger.info("------------std------------")
    classify(['std_data/std_data'], 'ACC_N', 'Gyro_z_Fil', [])
    '''


def run_gnb():
    # logger.info("------------start of new run---GNB--with partial fit-------")
    # logger.info("------------p1 ------------")
    # gnb_classify(['percentage/1/vector_percentage_1'], 'DJump_SIG_I_x LapEnd', '99-Gyro_z_Fil', [])
    # logger.info("------------p1 mean------------")
    # gnb_classify(['percentage/1/vector_percentage_mean_1'], 'DJump_SIG_I_x LapEnd', '99-Gyro_z_Fil', [])
    # logger.info("------------p1 std------------")
    # gnb_classify(['percentage/1/vector_percentage_mean_std_1'], 'DJump_SIG_I_x LapEnd', '99-std_Gyro_z_Fil', [])
    #
    # logger.info("------------p2------------")
    # gnb_classify(['percentage/2/vector_percentage_2'], 'DJump_SIG_I_x LapEnd', '98-Gyro_z_Fil', [])
    # logger.info("------------p2 mean------------")
    # gnb_classify(['percentage/2/vector_percentage_mean_2'], 'DJump_SIG_I_x LapEnd', '98-Gyro_z_Fil', [])
    # logger.info("------------p2 std------------")
    # gnb_classify(['percentage/2/vector_percentage_mean_std_2'], 'DJump_SIG_I_x LapEnd', '98-std_Gyro_z_Fil', [])
    #
    # logger.info("------------p5 ------------")
    # gnb_classify(['percentage/5/vector_percentage_5'], 'DJump_SIG_I_x LapEnd', '95-Gyro_z_Fil', [])
    # logger.info("------------p5 mean------------")
    # gnb_classify(['percentage/5/vector_percentage_mean_5'], 'DJump_SIG_I_x LapEnd', '95-Gyro_z_Fil', [])
    # logger.info("------------p5 std------------")
    # gnb_classify(['percentage/5/vector_percentage_mean_std_5'], 'DJump_SIG_I_x LapEnd', '95-std_Gyro_z_Fil', [])
    #
    # logger.info("------------p10 ------------")
    # gnb_classify(['percentage/10/vector_percentage_10'], 'DJump_SIG_I_x LapEnd', '90-Gyro_z_Fil', [])
    # logger.info("------------p10 mean------------")
    # gnb_classify(['percentage/10/vector_percentage_mean_10'], 'DJump_SIG_I_x LapEnd', '90-Gyro_z_Fil', [])
    # logger.info("------------p10 std------------")
    # gnb_classify(['percentage/10/vector_percentage_mean_std_10'], 'DJump_SIG_I_x LapEnd', '90-std_Gyro_z_Fil', [])

    logger.info("------------p10 std----without processed--------")
    gnb_classify(['percentage/10/vector_percentage_mean_std_10'], '0-mean_ACC_N', '90-std_Gyro_z_Fil', [])
    logger.info("------------p10 std----only processed--------")
    gnb_classify(['percentage/10/vector_percentage_mean_std_10'], 'DJump_SIG_I_x LapEnd', 'DJump_Abs_I_z LapEnd', [])

    i = 0
    drops = []
    while i < 100:
        drops.append(str(i) + "-mean_ACC_N_ROT_filtered")
        drops.append(str(i) + "-std_ACC_N_ROT_filtered")
        i += 10
    logger.info("------------p10 std----without processed without n rot filtered--------")
    gnb_classify(['percentage/10/vector_percentage_mean_std_10'], '0-mean_ACC_N', '90-std_Gyro_z_Fil', [])


    # logger.info("------------p20 ------------")
    # gnb_classify(['percentage/20/vector_percentage_20'], 'DJump_SIG_I_x LapEnd', '80-Gyro_z_Fil', [])
    # logger.info("------------p20 mean------------")
    # gnb_classify(['percentage/20/vector_percentage_mean_20'], 'DJump_SIG_I_x LapEnd', '80-Gyro_z_Fil', [])
    # logger.info("------------p20 std------------")
    # gnb_classify(['percentage/20/vector_percentage_mean_std_20'], 'DJump_SIG_I_x LapEnd', '80-std_Gyro_z_Fil', [])
    # logger.info("------------p20 std-----only processed-------")
    # gnb_classify(['percentage/20/vector_percentage_mean_std_20'], 'DJump_SIG_I_x LapEnd', 'DJump_Abs_I_z LapEnd', [])
    #

    # logger.info("------------p20 std-----without processed, without n rot filtered-------")
    # gnb_classify(['percentage/20/vector_percentage_mean_std_20'], '0-mean_ACC_N', '80-std_Gyro_z_Fil', drops)

    # logger.info("------------p25 ------------")
    # gnb_classify(['percentage/25/vector_percentage_25'], 'DJump_SIG_I_x LapEnd', '75-Gyro_z_Fil', [])
    # logger.info("------------p25 mean------------")
    # gnb_classify(['percentage/25/vector_percentage_mean_25'], 'DJump_SIG_I_x LapEnd', '75-Gyro_z_Fil', [])
    # logger.info("------------p25 std------------")
    # gnb_classify(['percentage/25/vector_percentage_mean_std_25'], 'DJump_SIG_I_x LapEnd', '75-std_Gyro_z_Fil', [])

    # logger.info("------------avg ------------")
    # gnb_classify(['averaged_data/averaged_data'], 'ACC_N', 'DJump_Abs_I_z LapEnd', [])
    # logger.info("------------avg std------------")
    # gnb_classify(['avg_std_data/avg_std_data'], 'avg_ACC_N', 'DJump_Abs_I_z LapEnd', [])
    # logger.info("------------std------------")
    # gnb_classify(['std_data/std_data'], 'ACC_N', 'Gyro_z_Fil', [])


if __name__ == '__main__':
    run_svc()