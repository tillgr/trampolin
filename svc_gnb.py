import shap
import pandas as pd
from holoviews.plotting.util import process_cmap
from pandas import DataFrame
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from os import listdir
from os.path import isfile, join

from neural_networks import bar_plots
from random_classifier import metrics
from matplotlib.colors import ListedColormap
import pickle

"""
This script is used to train, test, explain the Support Vector Classifiers and Gaussian Naive Bayes Classifiers.
"""

logging.basicConfig(filename='svc_gnb.log', format='%(asctime)s[%(name)s] - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('COMMON')
gnb_logger = logging.getLogger("GNB")
svc_logger = logging.getLogger("SVC")


def prediction_and_evaluate(classifier, testing_sample: DataFrame, test_targets: DataFrame):
    """

    Predict and evaluate a data set

    Parameters
    ----------
    classifier: classifier built with training data
    testing_sample: DataFrame - testing data (X)
    test_targets: DataFrame - testing target (y)

    """
    predicted = classifier.predict(testing_sample)
    score = accuracy_score(test_targets.to_numpy(), predicted)
    logger.info("prediction accuracy: " + str(score.round(4)))
    f1_average = 'weighted'
    f1 = f1_score(test_targets.to_numpy(), predicted, average=f1_average)
    logger.info("F1 score with param " + f1_average + " :" + str(f1.round(4)))
    mean_prec, mean_rec, mean_f, mean_youden = metrics(test_targets.to_numpy(), predicted)
    logger.info("Random classifier: Youden score: " + str(mean_youden.round(4)))
    logger.info("Random classifier: F1 score: " + str(mean_f.round(4)))


def get_samples_features(data: DataFrame, start_column: str, end_column: str):
    """

     preprocessed the data by selecting a feature range and lint the numbers to float.

     Parameters
     ----------
     data: DataFrame - data set
     start_column: str - start feature
     end_column: str - end feature

     :return: X: X of training or testing

     """
    X: DataFrame = data.loc[:, start_column:end_column]
    X.astype(dtype='float64')
    return X


def read_processed_data(filename: str):
    """

     read dataset from path and substitute "," with "." to avoid error by numbers

     Parameters
     ----------
     filename: str - the relative path to the dataset

     :return: processed_data: Dataframe
     """
    processed_data = pd.read_csv(filename)
    logger.info("read data set:" + filename)
    for column in processed_data.columns:
        try:
            processed_data[column] = processed_data[column].str.replace(',', '.')
        except AttributeError:
            pass
    return processed_data


def get_targets(data: DataFrame):
    """

     get targets of a dataset

     Parameters
     ----------
     data: DataFrame - dataset

     :return: data['Sprungtyp']: targets(y) of a dataset
     """
    targets = set(data['Sprungtyp'])
    return data['Sprungtyp']


def get_train_test_data(datasets: list):
    """

    get targets of a dataset

    Parameters
    ----------
    datasets: list - list of datasets in path form but without _train.csv or _test.csv

    :return: train, test: return the dataframes of train and test data

    """
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


def svc_classify(datasets: list, feature_start: str, feature_end: str, drops_keywords: list, reverse_drop: bool):
    """

    Classify with Support Vector Machine

    Parameters
    ----------
    datasets: list - list of datasets in path form but without _train.csv or _test.csv
    feature_start: str - start feature
    feature_end: str - end feature
    drops_keywords: list - the features will be drop if it has the keywords in this list.
    reverse_drop: bool - Set to True to drop the features does not contains the keywords.

    """
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
    test_targets = get_targets(test)
    clf_linear = SVC(kernel='linear')
    clf_linear.fit(X, y)
    prediction_and_evaluate(clf_linear, get_samples_features(test, feature_start, feature_end), test_targets)


def gnb_classify(datasets: list, feature_start: str, feature_end: str, drops_keywords: list, reverse_drop: bool):
    """

    Classify with Gaussian Naive Bayes

    Parameters
    ----------
    datasets: list - list of datasets in path form but without _train.csv or _test.csv
    feature_start: str - start feature
    feature_end: str - end feature
    drops_keywords: list - the features will be drop if it has the keywords in this list.
    reverse_drop: bool - Set to True to drop the features does not contains the keywords.

    """
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
    test_targets = get_targets(test)
    gnb.fit(X, y)
    prediction_and_evaluate(gnb, get_samples_features(test, feature_start, feature_end), test_targets)


def sample_x_test(x_test, y_test, num):
    """
    Samples data by retrieving only a certain number of each jump.

    Parameters
    ----------
    x_test : pandas.Dataframe - can be x_test and x_train
    y_test : pandas.Dataframe - can be y_test and y_train
    num : int - number of each jump to retrieve

    :return: sampled data Dataframe
    """

    df = x_test.copy()
    df['Sprungtyp'] = y_test
    counts = df['Sprungtyp'].value_counts()
    counts = counts.where(counts < num, num)
    x = pd.DataFrame(columns=df.columns)

    for jump in df['Sprungtyp'].unique():
        subframe = df[df['Sprungtyp'] == jump]
        x = x.append(subframe.sample(counts[jump], random_state=1), ignore_index=True)

    x = x.sample(frac=1, random_state=1)        # shuffle
    y = x['Sprungtyp']
    y = y.reset_index(drop=True)
    x = x.drop(['Sprungtyp'], axis=1)
    for column in x.columns:
        x[column] = x[column].astype(float).round(3)
    return x, y


def explain_model(train_data: str, test_data: str, feature_start: str, feature_end: str, drops_keywords: list, reverse_drop: bool, output_folder: str, classifier: str,
                  title: str, load_shap_from_disk):
    """

    Explain the built model by shap values in form of plots.

    Parameters
    ----------
    train_data: str - path of the train data set
    test_data: str - path of the train data set
    feature_start: str - start feature
    feature_end: str - end feature
    drops_keywords: list - the features will be drop if it has the keywords in this list.
    reverse_drop: bool - Set to True to drop the features does not contains the keywords.
    output_folder: str - output folder for bar plots
    classifier: str - Classifier type, "SVC" or "GNB"

    """
    train = pd.read_csv("Sprungdaten_processed" + train_data)
    test = pd.read_csv("Sprungdaten_processed" + test_data)

    drops = [col for col in train.columns if any(keyword in col for keyword in drops_keywords)]
    if reverse_drop:
        drops = [x for x in train.columns.tolist()[2:] if x not in drops]

    train = train.drop(columns=drops)
    test = test.drop(columns=drops)

    if feature_start == "":
        feature_start = train.columns.tolist()[2]
    if feature_end == "":
        feature_end = train.columns.tolist()[-1]

    X = get_samples_features(train, feature_start, feature_end)
    y = get_targets(train)
    y_test = get_targets(test)
    if classifier == "SVC":
        clf = SVC(kernel='linear')
    elif classifier == "GNB":
        clf = GaussianNB()
    else:
        raise RuntimeError("Invalid classifier type:" + classifier)

    clf.fit(X, y)
    X_test = get_samples_features(test, feature_start, feature_end)

    create_confision_matrix(X_test, y_test, clf, output_folder, classifier, title)

    cmap = ['#393b79','#5254a3','#6b6ecf','#9c9ede','#637939','#8ca252','#b5cf6b','#cedb9c','#8c6d31','#bd9e39','#e7ba52',
     '#e7cb94','#843c39','#ad494a','#d6616b','#e7969c','#7b4173','#a55194','#ce6dbd','#de9ed6','#3182bd','#6baed6',
     '#9ecae1','#c6dbef','#e6550d','#fd8d3c','#fdae6b','#fdd0a2','#31a354','#74c476','#a1d99b','#c7e9c0','#756bb1',
     '#9e9ac8','#bcbddc','#dadaeb','#636363','#969696','#969696','#d9d9d9','#f0027f','#f781bf','#f7b6d2','#fccde5',
     '#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']

    shap.initjs()

    shap_x_train, shap_y_train = sample_x_test(X, y, 3)
    shap_x_test, shap_y_test = sample_x_test(X_test, y_test, 6)

    if classifier == "SVC":
        explainer = shap.KernelExplainer(clf.decision_function, shap_x_train)
    elif classifier == "GNB":
        explainer = shap.KernelExplainer(clf.predict_proba, shap_x_train)
    else:
        raise RuntimeError("Invalid classifier type:" + classifier)

    if not load_shap_from_disk:
        shap_values = explainer.shap_values(shap_x_test)
        with open(output_folder + 'shap_data.pkl', 'wb') as f:
            pickle.dump([shap_values, shap_x_train, shap_y_train, shap_x_test, shap_y_test], f)
    else:
        with open(output_folder + "shap_data.pkl", 'rb') as f:
            shaps = pickle.load(f)
            shap_values = shaps[0]


    bar_plots(shap_values, shap_x_test, shap_y_test, bar='percentual', folder=output_folder, save_data=output_folder, size=(55, 30))
    bar_plots(shap_values, shap_x_test, shap_y_test, bar='summary', folder=output_folder, save_data=output_folder, size=(55, 30))
    bar_plots(shap_values, shap_x_test, shap_y_test, save_data=output_folder,
              bar='percentual', size=(50, 30),
              jumps=['Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C', 'Schraubensalto',
                     'Schraubensalto A', 'Schraubensalto C', 'Doppelsalto B', 'Doppelsalto C'],
              name='Saltos')

    shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(25, 20), color=ListedColormap(cmap), class_names=shap_y_test.unique(), max_display=20)
    shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(25, 20), color=ListedColormap(cmap), class_names=shap_y_test.unique(), max_display=68)
    saltoA = np.where(shap_y_test.unique() == 'Salto A')[0][0]
    shap.summary_plot(shap_values[saltoA], shap_x_test, plot_size=(25, 15), title='Salto A')
    saltoB = np.where(shap_y_test.unique() == 'Salto B')[0][0]
    shap.summary_plot(shap_values[saltoB], shap_x_test, plot_size=(25, 15), title='Salto B')
    saltoC = np.where(shap_y_test.unique() == 'Salto C')[0][0]
    shap.summary_plot(shap_values[saltoC], shap_x_test, plot_size=(25, 15), title='Salto C')


def create_confision_matrix(X_test, y_test, clf, output_folder, classifier, title):
    """

    create confusion matrix.

    Parameters
    ----------
    X_test: Dataframe - X of the test dataset
    y_test: Series - y of the test dataset
    clf: - trained classifier

    """
    y_pred = clf.predict(X_test)
    cmap_cm = process_cmap('summer')
    cmap_cm.insert(0, '#ffffff')
    cmap_cm.insert(-1, '#000000')
    cmap_cm = ListedColormap(cmap_cm)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    pd.DataFrame(cm, columns=y_test.unique(), index=y_test.unique()).to_csv(output_folder + "confusion_matrix.csv")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap=cmap_cm)
    disp.figure_.set_figwidth(35)
    disp.figure_.set_figheight(25)
    disp.figure_.autofmt_xdate()
    plt.title(classifier + title)
    plt.tight_layout()
    plt.savefig(output_folder + classifier+"_" + title + '_ConfusionMatrix.png')



def collect_all_data_sets(folder: str, data_sets: set):
    """

    Help function to get the datasets iteratively

    Parameters
    ----------
    folder: str - root folder for the the data sets
    data_sets: set - a set object to collect the data sets iteratively

    :return: data_sets - a set of dataset filepath from the given folder

    """
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
    """

    Help function to run SVC on all data sets

    Parameters
    ----------
    folder: str - root folder for the the data sets
    drops: list - the features that should be drop

    """

    data_sets = set()
    collect_all_data_sets(folder, data_sets)
    for ds in data_sets:
        svc_classify([ds], "", "", drops, True)


def run_gnb_auto(folder: str, drops: list):
    """

    Help function to run GNB on all data sets

    Parameters
    ----------
    folder: str - root folder for the the data sets
    drops: list - the features that should be drop

    """
    data_sets = set()
    collect_all_data_sets(folder, data_sets)
    for ds in data_sets:
        gnb_classify([ds], "", "", drops, False)


def prepare_data(data_train, data_test, pp_list):
    first_djumps = set([col for col in data_train.columns if 'DJump' in col]) - set(
        [col for col in data_train.columns if 'DJump_SIG_I_S' in col]) \
                   - set([col for col in data_train.columns if 'DJump_ABS_I_S' in col]) - set(
        [col for col in data_train.columns if 'DJump_I_ABS_S' in col])
    if 1 not in pp_list:
        data_train = data_train.drop(first_djumps, axis=1)
        data_test = data_test.drop(first_djumps, axis=1)
    if 2 not in pp_list:
        data_train = data_train.drop([col for col in data_train.columns if 'DJump_SIG_I_S' in col], axis=1)
        data_test = data_test.drop([col for col in data_test.columns if 'DJump_SIG_I_S' in col], axis=1)
    if 3 not in pp_list:
        data_train = data_train.drop([col for col in data_train.columns if 'DJump_ABS_I_S' in col], axis=1)
        data_test = data_test.drop([col for col in data_test.columns if 'DJump_ABS_I_S' in col], axis=1)
    if 4 not in pp_list:
        data_train = data_train.drop([col for col in data_train.columns if 'DJump_I_ABS_S' in col], axis=1)
        data_test = data_test.drop([col for col in data_test.columns if 'DJump_I_ABS_S' in col], axis=1)

    X_train = data_train.drop('Sprungtyp', axis=1)
    X_train = X_train.drop(['SprungID'], axis=1)
    X_test = data_test.drop('Sprungtyp', axis=1)
    X_test = X_test.drop(['SprungID'], axis=1)

    y_train = data_train['Sprungtyp']
    y_test = data_test['Sprungtyp']

    return X_train, y_train, X_test, y_test


def jump_core_detection(classifier: str, datasets, pp_list, title, jump_length=0):
    """
    Trains many different models with differently cut data. We cut from back to front, front to back, and from both sides

    Parameters
    ----------
    classifier : str
        'SVC' or 'GNB'
    data_train : pandas.Dataframe
        dataframe read from .csv file
    data_test : pandas.Dataframe
        dataframe read from .csv file
    pp_list : list
        a list with values from 1 to 4: [1, 2, 3, 4]. Corresponds to the blocks of preprocessed data. 1: first 9 columns, 2, 3, 4: 12 columns each
    title : str
        title of the plot
    :return: dictionary with scores of all trained models
    """
    data_train, data_test = get_train_test_data(datasets)
    scores = {}
    percentage = int(100 / jump_length)
    full_list = [l for l in range(0, 100, percentage)]

    variants = []
    for i in range(jump_length - 1):
        variants.append([l for l in range(i + 1)])
    for i in range(1, jump_length):
        variants.append(list(range(i, jump_length)))
    for i in range(int((jump_length - 1) / 2)):
        both_sides = list(set(list(range(jump_length))) - set(list(range(0, jump_length))[i + 1:-1 - i]))
        both_sides.sort()
        variants.append(both_sides)
    print(variants)

    for variant in variants:
        indexes = []
        for i in range(int(len(data_train) / jump_length)):
            for to_delete in variant:
                indexes.append(i * jump_length + to_delete)
        data_train_copy = data_train.drop(indexes)
        print(data_train_copy)
        indexes = []
        for i in range(int(len(data_test) / jump_length)):
            for to_delete in variant:
                indexes.append(i * jump_length + to_delete)
        data_test_copy = data_test.drop(indexes)
        X_train, y_train, X_test, y_test = prepare_data(data_train_copy, data_test_copy, pp_list)
        if classifier == "SVC":
            clf = SVC(kernel='linear')
        elif classifier == "GNB":
            clf = GaussianNB()
        else:
            raise RuntimeError("Invalid classifier type:" + classifier)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Accuracy score:  {str(accuracy_score(y_test, y_pred).__round__(4))}")
        score = accuracy_score(y_test, y_pred).__round__(4)
        print(score)

        variant_output = [v * percentage for v in variant]
        variant_output = list(set(full_list) - set(variant_output))
        variant_output.sort()

        scores[str(variant_output[0]) + ' - ' + str(variant_output[-1])] = score

    print(f"scores:  {scores}")

    min_y_value = 70
    plt.figure(figsize=(13, 13))
    plt.suptitle(title)
    plt.xlabel('Data')
    plt.ylabel('Accuracy')
    plt.axis([0, full_list[-1], min_y_value, 100])
    plt.xticks(range(0, 100 + percentage, percentage))
    plt.yticks(range(min_y_value, 105, 5))
    plt.grid(True, axis='x')
    # cmap = process_cmap('brg', len(scores))

    for i in range(len(scores)):
        entry = list(scores.items())[i]
        start, end = entry[0].split('-')
        acc = entry[1] * 100
        print(acc)
        print(min_y_value)
        if int(acc) >= min_y_value:
            if start.replace(' ', '') == '0':
                plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#0000ff', alpha=0.7)
            elif end.replace(' ', '') == str(full_list[-1]):
                plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#ff0000', alpha=0.7)
            else:
                plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#00ff00', alpha=0.7)
    plt.show()


if __name__ == '__main__':
    drops = []
    train = "/without_preprocessed/percentage/25/vector_percentage_mean_std_25_train.csv"
    test = "/without_preprocessed/percentage/25/vector_AJ_percentage_mean_std_25.csv"
    output_folder = 'plots/SVC/without_preprocessed/AJ/'
    explain_model(train, test, "", "", [], False, output_folder, "SVC", "without_vector_percentage_mean_25", True)
    #datasets = ["Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10"]
    #title = 'GNB with pp: percentage_mean_std_10'
    #jump_core_detection("GNB", datasets, [1, 2, 3, 4], title, 10)


