import numpy as np
import pandas as pd
import os
import openpyxl
import random
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    name = "data_point_jumps"
    trainings_data = pd.read_csv("Sprungdaten_processed/" + name + "/" + name + "_train.csv")
    test_data = pd.read_csv('Sprungdaten_processed/' + name + '/' + name + '_test.csv')
    classify(trainings_data, test_data, 10000)
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
        tp, fp, tn, fn = metrics(label, c)
        print(tp, fp, fn, tn)
        # Precision
        prec = tp/(tp+fp)
        # Recall
        rec = tp/(fn+tp)
        # F1 Score
        f = 2*(prec*rec) / (prec + rec)
        # Youden
        youden = rec + (tn/(tn + fp)) - 1

    else:
        acc = []
        prec = []
        rec = []
        youden = []
        f = []
        best = 0
        for i in range(loops):
            print(str(i) + '/' + str(loops))
            c = random.sample(r, len(test_data['SprungID'].unique()))
            a = accuracy_score(label, c)
            acc.append(a)
            tp, fp, tn, fn = metrics(label, c)
            prec.append((tp/(tp+fp)))
            rec.append((tp/(fn+tp)))
            f.append((2*(prec[-1] * rec[-1]) / (prec[-1] + rec[-1])))
            y = (rec[-1] + (tn/(tn + fp)) - 1)
            youden.append(y)
            if y > best:
                best = y
        acc = sum(acc) / len(acc)
        prec = sum(prec) / len(prec)
        rec = sum(rec) / len(rec)
        f = sum(f) / len(f)
        youden = sum(youden) / len(youden)
        print('best Youden:' + str(best))
    print("Accuracy: " + str(acc))
    print("Youden: " + str(youden))
    print("F1-Score: " + str(f))
    print("Precision: " + str(prec))
    print("Recall: " + str(rec))
    return acc


def metrics(label, c):
    conf = confusion_matrix(label, c)
    tp = sum(conf.diagonal())
    fp = 0
    fn = 0
    tn = 0
    for r in range(len(conf)):
        # fp
        row = conf[r]
        row = np.delete(row, r)
        fp += sum(row)
        # fn
        col = conf[:, r]
        col = np.delete(col, r)
        fn += sum(col)
        # tn
        sub_array = np.delete(conf, r, 0)
        sub_array = np.delete(sub_array, r, 1)
        tn += sum(sum(sub_array))
    return tp, fp, tn, fn


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
