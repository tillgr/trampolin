import numpy as np
import pandas as pd
import os
import openpyxl
import random
from sklearn.metrics import accuracy_score, confusion_matrix
import sklearn as sk


def main():
    name = "data_point_jumps"
    trainings_data = pd.read_csv("Sprungdaten_processed/" + name + "/" + name + "_train.csv")
    test_data = pd.read_csv('Sprungdaten_processed/' + name + '/' + name + '_test.csv')
    classify(trainings_data, test_data, 100)
    return


def classify(trainings_data, test_data, loops=None):
    dis = distribution(trainings_data).to_frame()
    dis = dis.reset_index()
    dis.columns = ['Sprungtyp', 'Anzahl']
    r = []
    for i in range(len(dis)):
        temp = [dis['Sprungtyp'][i]]
        temp = temp * dis['Anzahl'][i]
        r.extend(temp)
    random.shuffle(r)
    label = get_target(test_data)
    if loops is None:
        c = random.sample(r, len(test_data['SprungID'].unique()))

        acc = accuracy_score(label, c)
        prec, rec, f, youden = metrics(label, c)

    else:
        acc = []
        prec_list = []
        rec_list = []
        youden_list = []
        f_list = []
        best = 0
        for i in range(loops):
            print(str(i) + '/' + str(loops))
            c = random.sample(r, len(test_data['SprungID'].unique()))
            a = accuracy_score(label, c)
            acc.append(a)
            prec, rec, f, youden = metrics(label, c)
            prec_list.append(prec)
            rec_list.append(rec)
            f_list.append(f)
            youden_list.append(youden)
            if youden > best:
                best = youden
        acc = sum(acc) / len(acc)
        prec = sum(prec_list) / len(prec_list)
        rec = sum(rec_list) / len(rec_list)
        f = sum(f_list) / len(f_list)
        youden = sum(youden_list) / len(youden_list)
        print('best Youden:' + str(best))
    print("Accuracy: " + str(acc))
    print("Youden: " + str(youden))
    print("F1-Score: " + str(f))
    print("Precision: " + str(prec))
    print("Recall: " + str(rec))
    return acc


def metrics(label, c):

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

    # print("prec = %f, rec = %f, f-score = %f, youden = %f"%(mean_prec, mean_rec, mean_f, mean_youden))

    return mean_prec, mean_rec, mean_f, mean_youden


def get_target(test_data):
    labels = []
    for jump in test_data['SprungID'].unique():
        labels.append(test_data.loc[test_data['SprungID'] == jump]['Sprungtyp'].values[0])
    return labels


def distribution(trainings_data):
    # creates the distribution from df
    # returns df
    jumps = trainings_data['SprungID'].unique()
    counter = {}
    i = 0
    df = pd.DataFrame(columns=['Sprungtyp'])
    for jump in jumps:
        print(i)
        i += 1
        temp = trainings_data.loc[trainings_data['SprungID'] == jump]
        df = df.append({'Sprungtyp': temp['Sprungtyp'].values[0]}, ignore_index=True)
        """
        if temp['Sprungtyp'].values[0] in counter.keys():
            counter[temp['Sprungtyp'].values[0]] += 1
        else:
            counter[temp['Sprungtyp'].values[0]] = 1


    print(counter)"""
    print(df['Sprungtyp'].value_counts())
    return df['Sprungtyp'].value_counts()


def save_as_xlsx(dis, name):
    # save the distribution df as an xlsx (needed for method classify)
    # need file name ( recommendation: name include name from cvs file)
    dis.to_excel(os.getcwd() + '/Sprungdaten_processed/' + name + ".xlsx")


if __name__ == '__main__':
    main()
