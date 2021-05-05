import numpy as np
import pandas as pd
import os
import openpyxl
import random
from sklearn.metrics import accuracy_score


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
    else:
        acc = []
        best = 0
        for i in range(loops):
            c = random.sample(r, len(test_data['SprungID'].unique()))
            a = accuracy_score(label, c)
            acc.append(a)
            if a > best:
                best = a
        acc = sum(acc) / len(acc)
        print('best Acc:' + str(best))
    print("Acc: " + str(acc))
    return acc


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
