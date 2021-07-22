import numpy as np
import pandas as pd
import shap
from pandas import DataFrame
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from random_classifier import metrics
from holoviews.plotting.util import process_cmap
from matplotlib.colors import ListedColormap
from neural_networks import bar_plots
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import pickle


def prepare_data(data_train, data_test, pp_list):
    """

    Prepare data for use in Classifier

    Parameters
    ----------
    data_train : pandas.Dataframe
        dataframe read from .csv file
    data_test : pandas.Dataframe
        dataframe read from .csv file
    pp_list : list
        a list with values from 1 to 4: [1, 2, 3, 4]. Corresponds to the blocks of preprocessed data. 1: first 9 columns,
        2, 3, 4: 12 columns each

    :return: X_train: - features of the train data set,
             y_train: - targets of the train data set,
             X_test: - features of the test data set,
             y_test: - targets of the test data set
    """

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

    x_train = data_train.drop('Sprungtyp', axis=1)
    x_train = x_train.drop(['SprungID'], axis=1)
    x_test = data_test.drop('Sprungtyp', axis=1)
    x_test = x_test.drop(['SprungID'], axis=1)

    if 1 in pp_list and 2 in pp_list and 3 in pp_list and 4 in pp_list and "only" in pp_list:
        x_train = x_train.drop([col for col in x_train.columns if 'DJump' not in col], axis=1)
        x_test = x_test.drop([col for col in x_test.columns if 'DJump' not in col], axis=1)

    y_train = data_train['Sprungtyp']
    y_test = data_test['Sprungtyp']

    return x_train, y_train, x_test, y_test


def all_parameters_classifier(data_train, data_test, pp_list, accuracy=0):
    """

    Classifier with all possible parameters and with accuracy

    Parameters
    ----------
    data_train : pandas.Dataframe
        dataframe read from .csv file
    data_test : pandas.Dataframe
        dataframe read from .csv file
    pp_list : list
        a list with values from 1 to 4: [1, 2, 3, 4]. Corresponds to the blocks of preprocessed data. 1: first 9 columns,
        2, 3, 4: 12 columns each
    accuracy: float
        accuracy for the Classifier

    :return:

    """
    X_train, y_train, X_test, y_test = prepare_data(data_train, data_test, pp_list)
    for losses in ['log', 'modified_huber', 'squared_hinge', 'perceptron']:
        for penalty in ['l2', 'l1', 'elasticnet']:
            for maxi in [10000]:
                clf = SGDClassifier(loss=losses, penalty=penalty, alpha=0.0001,
                                    l1_ratio=0.15, fit_intercept=True, max_iter=maxi,
                                    tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                                    n_jobs=None, random_state=10, learning_rate='optimal',
                                    eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                                    n_iter_no_change=5, class_weight=None,
                                    warm_start=False, average=False).fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                if accuracy_score(y_test, y_pred) > accuracy:
                    print(
                        f"Accuracy score: {losses} , {penalty} , {maxi}:  {str(accuracy_score(y_test, y_pred).__round__(4))}")
                    mean_prec, mean_rec, mean_f, mean_youden = metrics(y_test, y_pred)
                    print(f"Accuracy youden score: {str(mean_youden.round(4))}")
                    print(f"Accuracy f1 score: {str(mean_f.round(4))}")


def all_datasets_with_all_parameters_classifier(pp_list, accuracy=0):
    """

    Classifier with all possible parameters and all data sets

    Parameters
    ----------
    pp_list: list
        list of DJumps, that should be included in train and test data sets
    accuracy: float
        accuracy for the Classifier

    :return:

    """
    for i in [1, 2, 5, 10, 20, 25]:
        for calc_type in ['', 'mean_', 'mean_std_']:
            print(f"Folder: {i}")
            print(f"> Type: {calc_type}")
            data_train = pd.read_csv("../Sprungdaten_processed/with_preprocessed/percentage/" + str(
                i) + "/vector_percentage_" + calc_type + str(i) + "_train.csv")
            data_test = pd.read_csv("../Sprungdaten_processed/with_preprocessed/percentage/" + str(
                i) + "/vector_percentage_" + calc_type + str(i) + "_test.csv")

            all_parameters_classifier(data_train, data_test, pp_list, accuracy)

    for type in ['averaged_data/averaged', 'avg_std_data/avg_std', 'std_data/std']:
        print(f"> Type: {type}")
        data_train = pd.read_csv("../Sprungdaten_processed/with_preprocessed/" + str(
            type) + "_data_train.csv")
        data_test = pd.read_csv("../Sprungdaten_processed/with_preprocessed/" + str(
            type) + "_data_test.csv")
        all_parameters_classifier(data_train, data_test, pp_list, accuracy)


def get_best_data_set_with_preprocessed():
    """
    train and test data sets with best accuracy for preprocessed data

    :return: train_data - train data set with preprocessed data and best accuracy
             test_data - test data set with preprocessed data and best accuracy

    """
    train_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')
    train_data = train_data.drop([col for col in train_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    test_data = test_data.drop([col for col in test_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    return train_data, test_data


def get_best_data_set_without_preprocessed():
    """
    train and test data sets with best accuracy and without preprocessed data

    :return: train_data - train data set without preprocessed data and best accuracy
             test_data - test data set without preprocessed data and best accuracy

    """
    train_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')
    return train_data, test_data


def get_targets(data):
    """

    Get targets of the data set

    Parameters
    ----------
    data: pandas.DataFrame

    :return: data['Sprungtyp'] - targets of the data set

    """
    return data['Sprungtyp']


def sample_x_test(x_test, y_test, num):
    """

    Samples data by retrieving only a certain number of each jump.

    Parameters
    ----------
    :param x_test : pandas.Dataframe
        can be x_test and x_train
    :param y_test : pandas.Dataframe
        can be y_test and y_train
    :param num : int
        number of each jump to retrieve

    :return: sampled data Dataframe

    """
    df = x_test.copy()
    df['Sprungtyp'] = y_test
    counts = df['Sprungtyp'].value_counts()
    counts = counts.where(counts < num, other=num)
    x = pd.DataFrame(columns=df.columns)

    for jump in df['Sprungtyp'].unique():
        subframe = df[df['Sprungtyp'] == jump]
        x = x.append(subframe.sample(counts[jump], random_state=1), ignore_index=True)

    x = x.sample(frac=1)  # shuffle
    y = x['Sprungtyp']
    y = y.reset_index(drop=True)
    x = x.drop(['Sprungtyp'], axis=1)
    for column in x.columns:
        x[column] = x[column].astype(float).round(3)

    return x, y


def sgd_classifier(X_train, y_train, X_test, y_test, loss, penalty, max_iter):
    """
    SGD Classifier with specific parameters

    Parameters
    ----------
    X_train: pandas.Dataframe
        features of the train data set,
    y_train: pandas.Dataframe
        targets of the train data set,
    X_test: pandas.Dataframe
        features of the test data set,
    y_test: pandas.Dataframe
        targets of the test data set
    loss: str
        classifier parameter
    penalty: str
        classifier parameter
    max_iter: int
        classifier parameter

    :return: clf - sgd Classifier
             y_pred - predicted class labels for the provided data

    """

    clf = SGDClassifier(loss=loss, penalty=penalty, alpha=0.0001,
                        l1_ratio=0.15, fit_intercept=True, max_iter=max_iter,
                        tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                        n_jobs=None, random_state=10, learning_rate='optimal',
                        eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                        n_iter_no_change=5, class_weight=None,
                        warm_start=False, average=False).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(
        f"Accuracy score: {loss} , {penalty} , {max_iter}:  {str(accuracy_score(y_test, y_pred).__round__(4))}")
    mean_prec, mean_rec, mean_f, mean_youden = metrics(y_test, y_pred)
    print(f"Accuracy youden score: {str(mean_youden.round(4))}")
    print(f"Accuracy f1 score: {str(mean_f.round(4))}")
    return clf, y_pred


def shap_plots(data_train, data_test, pp_list, loss, penalty, max_iter, aj=None):
    """

    creates plots:
        1. Confusion Matrix
        2. Percentual Plot
        3. Percentual Saltos Plot
        4. Summary Plot
        5. Salto A,B,C

    creates data files:
        1. Confusion Matrix (csv)
        2. Percentual Plot (txt)
        3. Saltos Plot (txt)
        4. Common Shap Data (pkl)

    Parameters
    ----------
    data_train : pandas.Dataframe
        dataframe read from .csv file
    data_test : pandas.Dataframe
        dataframe read from .csv file
    pp_list : list
        a list with values from 1 to 4: [1, 2, 3, 4]. Corresponds to the blocks of preprocessed data. 1: first 9 columns,
        2, 3, 4: 12 columns each
    loss: str
        classifier parameter
    penalty: str
        classifier parameter
    max_iter: int
        classifier parameter
    aj: str
        Set for AJ-Data, changes appearance of the confusion matrix

    :return:

    """
    X_train, y_train, X_test, y_test = prepare_data(data_train, data_test, pp_list)
    clf, y_pred = sgd_classifier(X_train, y_train, X_test, y_test, loss, penalty, max_iter)

    if aj is None:
        cmap_cm = process_cmap('summer')
        cmap_cm.insert(0, '#ffffff')
        cmap_cm.insert(-1, '#000000')
        cmap_cm = ListedColormap(cmap_cm)

        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        pd.DataFrame(cm, columns=y_test.unique(), index=y_test.unique()).to_csv(
            '../plots/SGD/without_preprocessed/confusion_matrix.csv')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(cmap=cmap_cm)
        disp.figure_.set_figwidth(35)
        disp.figure_.set_figheight(25)
        disp.figure_.autofmt_xdate()
        plt.tick_params(axis='x', labelsize=10, labelrotation=45, grid_linewidth=5)
        plt.title("SGD_without_mean_std_10_ConfusionMatrix")
        plt.tight_layout()
        plt.savefig('../plots/SGD/without_preprocessed/confusionMatrix')
        plt.show()

    else:
        cmap_cm_AJ = ['#ffffff', '#048166']
        cmap_cm_AJ = ListedColormap(cmap_cm_AJ)

        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        pd.DataFrame(cm, columns=y_test.unique(), index=y_test.unique()).to_csv(
            '../plots/SGD/without_preprocessed/AJ/confusion_matrix.csv')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(cmap=cmap_cm_AJ)
        disp.figure_.set_figwidth(35)
        disp.figure_.set_figheight(25)
        disp.figure_.autofmt_xdate()
        plt.tick_params(axis='x', labelsize=10, labelrotation=45, grid_linewidth=5)
        plt.title("SGD_without_mean_std_10_ConfusionMatrix_AJ")
        plt.tight_layout()
        plt.savefig('../plots/SGD/without_preprocessed/AJ/confusionMatrix_AJ')
        plt.show()

    shap_x_test, shap_y_test = sample_x_test(X_test, y_test, 3)
    shap_x_train, shap_y_train = sample_x_test(X_train, y_train, 6)

    explainer = shap.KernelExplainer(clf.decision_function, shap_x_train)
    shap_values = explainer.shap_values(shap_x_test)

    with open('../plots/SGD/without_preprocessed/' + 'shap_data.pkl', 'wb') as f:
        pickle.dump([shap_values, shap_x_train, shap_y_train, shap_x_test, shap_y_test], f)

    bar_plots(shap_values, shap_x_test, shap_y_test, folder='../plots/SGD/without_preprocessed/',
              bar='percentual', size=(50, 30))

    bar_plots(shap_values, shap_x_test, shap_y_test, folder='../plots/SGD/without_preprocessed/',
              bar='percentual', size=(50, 30),
              jumps=['Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C', 'Schraubensalto',
                     'Schraubensalto A', 'Schraubensalto C', 'Doppelsalto B', 'Doppelsalto C'],
              name='Saltos')

    bar_plots(shap_values, shap_x_test, shap_y_test, folder='../plots/SGD/with_preprocessed/',
              size=(30, 45), name='summary_without_preprocessed')

    saltoA = np.where(shap_y_test.unique() == 'Salto A')[0][0]
    shap.summary_plot(shap_values[saltoA], shap_x_test, plot_size=(30, 12), title='Salto A')
    saltoB = np.where(shap_y_test.unique() == 'Salto B')[0][0]
    shap.summary_plot(shap_values[saltoB], shap_x_test, plot_size=(30, 12), title='Salto B')
    saltoC = np.where(shap_y_test.unique() == 'Salto C')[0][0]
    shap.summary_plot(shap_values[saltoC], shap_x_test, plot_size=(30, 12), title='Salto C')


def jump_core_detection(data_train, data_test, pp_list):
    """
        Trains many different models with differently cut data. We cut from back to front, front to back, and from both sides

        Parameters
        ----------

        data_train : pandas.Dataframe
            dataframe read from .csv file
        data_test : pandas.Dataframe
            dataframe read from .csv file
        pp_list : list
            a list with values from 1 to 4: [1, 2, 3, 4]. Corresponds to the blocks of preprocessed data. 1: first 9 columns, 2, 3, 4: 12 columns each


        :return: dictionary with scores of all trained models
        """
    jump_length = int(len(list(data_test.drop([col for col in data_train.columns if 'DJump' in col], axis=1)
                               .drop(['SprungID', 'Sprungtyp'], axis=1).columns))
                      / len(
        [c for c in list(data_test.drop([col for col in data_train.columns if 'DJump' in col], axis=1)
                         .drop(['SprungID', 'Sprungtyp'], axis=1).columns) if c.startswith('0_')]))

    print(jump_length)

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
        clf = SGDClassifier(loss='log', penalty='l1', alpha=0.0001,
                            l1_ratio=0.15, fit_intercept=True, max_iter=10000,
                            tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                            n_jobs=None, random_state=10, learning_rate='optimal',
                            eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                            n_iter_no_change=5, class_weight=None,
                            warm_start=False, average=False).fit(X_train, y_train)
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
    plt.suptitle('SGD without pp: percentage_mean_std_10')
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
    train_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')
    #all_datasets_with_all_parameters_classifier([1, 2, 3, 4])
    #jump_core_detection(train_data, test_data, [1, 2, 3, 4])
    '''clf = SGDClassifier(loss='log', penalty='l1', alpha=0.0001,
                            l1_ratio=0.15, fit_intercept=True, max_iter=10000,
                            tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                            n_jobs=None, random_state=10, learning_rate='optimal',
                            eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                            n_iter_no_change=5, class_weight=None,
                            warm_start=False, average=False).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if accuracy_score(y_test, y_pred) > 0:
            print(f"Accuracy score:  {str(accuracy_score(y_test, y_pred).__round__(4))}")
            mean_prec, mean_rec, mean_f, mean_youden = metrics(y_test, y_pred)
            print(f"Accuracy youden score: {str(mean_youden.round(4))}")
            print(f"Accuracy f1 score: {str(mean_f.round(4))}")'''
    #shap_plots(train_data, test_data, [1, 2, 3, 4], 'perceptron', 'l1', 10000)
    prepare_data(train_data, test_data, [1, 2, 3, 4, "only"])
