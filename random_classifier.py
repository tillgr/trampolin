import numpy as np
import pandas as pd
import os
import openpyxl
import random


def main():
    trainings_data = pd.read_csv("Sprungdaten_processed/data_point_jumps.csv")

    dis = distribution(trainings_data)
    save_as_xlsx(dis, 'Verteilung_data_point_jumps')
    classify(os.getcwd() + "/Sprungdaten_processed/Verteilung_data_point_jumps.xlsx", 20)
    return


def classify(path_distribution, len_to_class):
    # need the distribution file from the data and how many items must be classify
    # returns a list with classification
    dis = pd.read_excel(path_distribution, engine="openpyxl")
    dis.columns = ['Sprungtyp', 'Anzahl']
    r = []
    for i in range(len(dis)):
        temp = [dis['Sprungtyp'][i]]
        temp = temp * dis['Anzahl'][i]
        r.extend(temp)
    random.shuffle(r)

    return random.sample(r, len_to_class)


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
