import numpy as np
import pandas as pd
import os
import openpyxl
import random
from sklearn.metrics import accuracy_score, confusion_matrix
import sklearn as sk


def main():
    # main method for naive classifier and generate distribution for data

    # for naive classifier
    pp = 'with_preprocessed'
    name = "percentage/25/percentage_25"
    trainings_data = pd.read_csv("Sprungdaten_processed/" + pp + "/" + name + "_train.csv")
    test_data = pd.read_csv('Sprungdaten_processed/' + pp + '/' + name + '_test.csv')
    acc, youden, f1 = classify(trainings_data, test_data, 'best', 10000)
    print('Accuracy = ' + str(round(acc, 10)))
    print('Youden = ' + str(round(youden, 10)))
    print('F1 = ' + str(round(f1, 10)))

    # for distribution
    data = pd.read_csv("Sprungdaten_processed/with_preprocessed/data_only_jumps.csv")
    dis = distribution(data)
    save_as_xlsx(dis, '/Sprungdaten_processed/', 'data_distribution_all_data')

    return


def classify(trainings_data, test_data, output='mean', loops=None):
    """
    random classfier, creates probabilities based on the distribution

    Parameters
    ----------
    trainings_data: Dataframe - train data set
    test_data: Dataframe - test data set
    output: str - 'mean' or 'best', default='mean';
        function will return best or mean Accuracy, Youden and F1 Score
    loops: int; how often the classifier should run, to get the best and mean score

    :return: acc - Accuracy, youden - Youden , f1 - F1 Score
    """
    dis = distribution(trainings_data).to_frame()
    dis = dis.reset_index()
    dis.columns = ['Sprungtyp', 'Anzahl']
    r = []
    for i in range(len(dis)):
        temp = [dis['Sprungtyp'][i]]
        temp = temp * dis['Anzahl'][i]
        r.extend(temp)
    random.shuffle(r)

    # get targets
    label = get_target(test_data)

    if loops is None:
        c = random.sample(r, len(test_data['SprungID'].unique()))

        acc = accuracy_score(label, c)
        prec, rec, f, youden = metrics(label, c)

        return acc, youden, prec

    else:
        acc = []
        prec_list = []
        rec_list = []
        youden_list = []
        f_list = []
        best_youden = 0
        print_progress_bar(0, loops, prefix='Progress:', suffix='Complete', length=50)
        for i in range(loops):
            c = random.sample(r, len(test_data['SprungID'].unique()))
            a = accuracy_score(label, c)
            acc.append(a)
            prec, rec, f, youden = metrics(label, c)
            prec_list.append(prec)
            rec_list.append(rec)
            f_list.append(f)
            youden_list.append(youden)
            if youden > best_youden:
                best_youden = youden
                best_acc = a
                best_f1 = f

            print_progress_bar(i+1, loops, prefix='Progress:', suffix='Complete', length=50)

        acc = sum(acc) / len(acc)
        prec = sum(prec_list) / len(prec_list)
        rec = sum(rec_list) / len(rec_list)
        f = sum(f_list) / len(f_list)
        youden = sum(youden_list) / len(youden_list)

    if output == 'mean':
        return acc, youden, f
    if output == 'best':
        return best_acc, best_youden, best_f1


def get_target(test_data):
    """
    getting from each jump the "Sprungtyp" and return this as a list

    Parameters
    ----------
    test_data: Dataframe - test data for creating distribution list

    :return: labels - list of jumps
    """
    labels = []
    for jump in test_data['SprungID'].unique():
        labels.append(test_data.loc[test_data['SprungID'] == jump]['Sprungtyp'].values[0])
    return labels


def metrics(label, c):
    """
    for getting Precision, Recall, F1  and Youden Score

    Parameters
    ----------
    label: list - y test (targets)
    c: list - predicted labels

    :return: mean_prec - Precision, mean_rec - Recall, mean_f - F1 Score, mean_youden - Youden Score
    """

    mcm = sk.metrics.multilabel_confusion_matrix(label, c)
    tn = mcm[:, 0, 0]
    fn = mcm[:, 1, 0]
    tp = mcm[:, 1, 1]
    fp = mcm[:, 0, 1]

    mean_prec = 0.0
    mean_rec = 0.0
    mean_f = 0.0
    mean_youden = 0.0

    num_classes = len(tp)

    for i in range(num_classes):
        prec = 0.0
        rec = 0.0

        # check for division by 0
        if (tp[i] + fp[i]) != 0:
            prec = tp[i] / (tp[i] + fp[i])
            mean_prec += prec
        if (fn[i] + tp[i]) != 0:
            rec = tp[i] / (fn[i] + tp[i])
            mean_rec += rec
        if (prec + rec) != 0:
            mean_f += 2 * (prec * rec) / (prec + rec)
        if (tn[i] + fp[i]) != 0:
            mean_youden += rec + (tn[i] / (tn[i] + fp[i])) - 1

    mean_prec /= num_classes
    mean_rec /= num_classes
    mean_f /= num_classes
    mean_youden /= num_classes

    return mean_prec, mean_rec, mean_f, mean_youden


def distribution(data):
    """
    creates a Series, of often each jump occur

    Parameters
    ----------
    data: Dataframe - Data for the distribution

    :return: Series with jump type as index
    """

    return data.groupby(['SprungID']).first()['Sprungtyp'].value_counts()


def save_as_xlsx(data, folder, name):
    """
    save the distribution df as an xlsx

    Parameters
    ----------
    data: Series or Dataframe - data that should be saved
    folder: str - path as String, where data should be saved
    name: str - name for the file

    :return:
    """

    data.to_excel(os.getcwd() + folder + name + ".xlsx")


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end ="\r"):
    """
    call in a loop to create terminal progress bar

    Parameters
    ----------
    iteration: int - current iteration (Required)
    total: int - total iterations(Required)
    prefix: str - prefix string (Optional)
    suffix: str - suffix string (Optional)
    decimals: int - positive number of decimals in percent complete (Optional)
    length: int - character length of bar (Optional)
    fill: str - bar fill character (Optional)
    print_end :str - end character (e.g. "\r", "\r\n")(Optional)

    :return:
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    main()
