import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import itertools

"""
#                                       #
#                                       #
#           Helper Functions            #
#                                       #
#                                       #
"""


def read_raw_csv_data(file):
    """
    Reads the data from the raw data csv files.
    It skips the first 2 rows to get only the data. For the columns it ready only the first row.

    :param file: Path to csv file -> e.g. "Sprungdaten Innotramp/.../Rohdaten...csv"
    :return: A Dataframe with the raw data
    """

    raw_data = pd.read_csv(file, header=None, skiprows=2, delimiter=";")
    raw_data.columns = pd.read_csv(file, nrows=0, delimiter=";").columns
    print(file)

    return raw_data


def read_xlsx_data(file):
    """
    Reads the data from an excel file.

    :param file: Path to xslx file -> e.g. "Sprungdaten Innotramp/.../Sprungzuordnung...xlsx"
    :return: A Dataframe with the raw data
    """

    raw_data = pd.read_excel(file, engine="openpyxl")
    print(file)

    return raw_data


def convert_to_numeric(x):
    """
    Function to apply to a dataframe. It removes the 'sec' and 'min' from the times so we can convert it to floats.

    :return: The Dataframe with normal time formatting
    """

    if "sec" in x:

        return float(x.replace(' sec', ''))

    if "min" in x:
        x = x.replace(" min", "")
        times = x.split(":")
        minutes = float(times[0])
        seconds = round(float(times[1]), 3)

        return float(minutes * 60.0 + seconds)


def convert_comma_to_dot(x):
    """
    Function to apply to a dataframe. It replaces all comma to dots so we can convert it to floats.

    :return: The Dataframe with normal number formatting
    """

    return float(x.replace(',', '.'))


def save_as_csv(data, name, folder=""):
    """
    Saves a dataset as csv.

    It automatically rounds all columns to 3 decimal points

    It always saves into Sprungdaten_processed/...

    :param data: dataset to be saved
    :param name: name of the dataset with .csv
    :param folder: folder after Sprungdaten_processed/  -> gets created if not already there

    """

    pd.options.mode.chained_assignment = None

    name_list = list(data.columns.values)
    name_list.remove('SprungID')
    name_list.remove('Sprungtyp')
    for column in name_list:
        try:
            data[column] = data[column].astype(float).round(3)
        except:
            print("couldnt round column " + column)

    if folder == "":
        data.to_csv('Sprungdaten_processed/' + name + ".csv", index=False)
    else:
        if not os.path.isdir('Sprungdaten_processed/' + folder):
            os.makedirs("Sprungdaten_processed/" + folder)
        data.to_csv('Sprungdaten_processed/' + folder + "/" + name + ".csv", index=False)


def sort_out_errors(data):
    """
    Sorts out all unwanted data and fixes the class names with spelling errors

    :param data: data to be fixed
    :return: fixed data
    """

    pd.options.mode.chained_assignment = None

    # delete all the unwanted data
    data = data.query('Sprungtyp != "Datenfehler" & Sprungtyp != "Fehlerhafte Daten" & Sprungtyp != "Unbekannt" & Sprungtyp != "Einturnen"')

    # delete where Sprungtyp == nan
    nan_value = float('NaN')
    data['Sprungtyp'].replace("", nan_value, inplace=True)
    data.dropna(subset=['Sprungtyp'], inplace=True)

    # delete jumps where DJumps all 0
    djumps = [col for col in data.columns if 'DJump' in col]
    all_data_djumps = data[djumps]
    data = data[(all_data_djumps == 0).sum(1) < len(djumps)]
    data = data.reset_index(drop=True)

    # correct spelling mistakes to make jump classes consistent
    data['Sprungtyp'].replace("Baby-Fliffis C", "Baby- Fliffis C", inplace=True)
    data['Sprungtyp'].replace("Voll ein Voll aus C", "Voll- ein- voll- aus C", inplace=True)
    data['Sprungtyp'].replace("Voll- ein- Voll- Aus B", "Voll- ein- voll- aus B", inplace=True)
    data['Sprungtyp'].replace("Voll ein Halb aus B", "Voll- ein- halb- aus B", inplace=True)
    data['Sprungtyp'].replace("Halb ein Halb aus B", "Halb- ein- halb- aus B", inplace=True)
    data['Sprungtyp'].replace("Halb ein Halb aus C", "Halb- ein- halb- aus C", inplace=True)
    data['Sprungtyp'].replace("Voll ein Voll aus C", "Voll- ein- voll- aus C", inplace=True)
    data['Sprungtyp'].replace("Voll ein Rudi aus B", "Voll- ein- Rudi- aus B", inplace=True)
    data['Sprungtyp'].replace("Voll ein Voll aus B", "Voll- ein- voll- aus B", inplace=True)
    data['Sprungtyp'].replace("Voll ein Voll aus A", "Voll- ein- voll- aus A", inplace=True)
    data['Sprungtyp'].replace("Voll ein Halb aus C", "Voll- ein- halb- aus C", inplace=True)
    data['Sprungtyp'].replace("Halb ein Rudy aus B", "Halb- ein- Rudy- aus B", inplace=True)
    data['Sprungtyp'].replace("Voll- ein- Voll- aus B", "Voll- ein- voll- aus B", inplace=True)
    data['Sprungtyp'].replace("Voll- ein- Voll- aus A", "Voll- ein- voll- aus A", inplace=True)
    data['Sprungtyp'].replace("Voll- ein- Voll- aus C", "Voll- ein- voll- aus C", inplace=True)
    data['Sprungtyp'].replace("Voll- ein - Doppel Aus A", "Voll- ein- doppel- aus A", inplace=True)
    data['Sprungtyp'].replace("Baby Fliffis C", "Baby- Fliffis C", inplace=True)
    data['Sprungtyp'].replace("Fliffis Rudi B", "Fliffis- Rudi B", inplace=True)
    data['Sprungtyp'].replace("51°", "Baby- Fliffis C", inplace=True)
    data['Sprungtyp'].replace("3/4 Salto Rw A", "3/4 Salto rw A", inplace=True)
    data['Sprungtyp'].replace("3/4 Salto vW A", "3/4 Salto vw A", inplace=True)
    data['Sprungtyp'].replace("3/4 Salto Vw  A", "3/4 Salto vw A", inplace=True)
    data['Sprungtyp'].replace("3/4 Salto Vw A", "3/4 Salto vw A", inplace=True)
    data['Sprungtyp'].replace("30/R", "3/4 Salto vw A", inplace=True)
    data['Sprungtyp'].replace("1 3/4 Salto Vw B", "1 3/4 Salto vw B", inplace=True)
    data['Sprungtyp'].replace("1 3/4 Salto Vw C", "1 3/4 Salto vw C", inplace=True)
    data['Sprungtyp'].replace("Voll ein 1 3/4 Salto Vw C", "Voll- ein 1 3/4 Salto vw C", inplace=True)
    data['Sprungtyp'].replace("801°", "Fliffis aus C", inplace=True)
    data['Sprungtyp'].replace("801<", "Fliffis aus B", inplace=True)
    data['Sprungtyp'].replace("Schraubensalto ", "Schraubensalto", inplace=True)
    data['Sprungtyp'].replace("Rudi ", "Rudi", inplace=True)
    data['Sprungtyp'].replace("Scharubensalto A", "Schraubensalto A", inplace=True)
    data['Sprungtyp'].replace("40<", "Salto rw B", inplace=True)
    data['Sprungtyp'].replace("40/", "Salto rw A", inplace=True)
    data['Sprungtyp'].replace("40°", "Salto rw C", inplace=True)
    data['Sprungtyp'].replace("41°", "Barani C", inplace=True)
    data['Sprungtyp'].replace("41/", "Barani A", inplace=True)
    data['Sprungtyp'].replace("43/", "Rudi", inplace=True)
    data['Sprungtyp'].replace("00V", "Grätschwinkel", inplace=True)
    data['Sprungtyp'].replace("1 3/4 salto Vorwärts B", "1 3/4 Salto vw B", inplace=True)
    data['Sprungtyp'].replace("Fliffis Rudi C", "Fliffis- Rudi C", inplace=True)

    return data


def split_train_test(data, test_size=0.2, min_num_jumps=2):
    """
    Split a given dataset into trainings- and test data. To keep the distribution consistent we stratify with Sprungtyp.

    We also drop all jumps which occur less than 2 times. As 2 isnt enough for the stratify algorithm.

    :param data: data to be split
    :param test_size: the fraction of the data to be the test set. default=0.2 -> 80 / 20 split
    :param min_num_jumps: the max number of occurences a jump can have and still be deleted. All jumps with min_num_jumps + 1 will be kept
    :return: train data and test data
    """

    # if the data consists of only one row per jump we can use the easy way
    if len(data['SprungID'].unique()) == len(data):
        # drop the jumps which occur <= 2 times
        indexes = np.where(data['Sprungtyp'].value_counts() == min_num_jumps)       # search for the first time a jump with too few occurences comes up
        one_time_jumps = data['Sprungtyp'].value_counts()[indexes[0][0]:].index     # drop all jumps after that

        for jump in one_time_jumps:
            index = data[data['Sprungtyp'] == jump].index
            data.drop(index, inplace=True)

        # split train - test -> 80 : 20 -> stratify to keep distribution of classes
        df_train, df_test, y_train, y_test = train_test_split(data, data['Sprungtyp'], stratify=data['Sprungtyp'],
                                                              test_size=test_size, random_state=1)
        print("Split of Train/Test Data complete")
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        return df_train, df_test

    # to keep the data of the jumps consistens and not let them split up, we need to do some more complicated operations
    df_split = pd.DataFrame(columns=['Sprungtyp', 'Data'])

    # put each jump in its own dataframe and concatenate those to df_split
    id_length = len(data['SprungID'].unique())
    for id in data['SprungID'].unique():
        print("Packing Dataframe: " + str(len(df_split) + 1) + "/" + str(id_length))
        jump = data[data['SprungID'] == id]
        df_split.loc[len(df_split)] = [jump['Sprungtyp'].unique()[0], jump]

    # drop the jumps which occur <= 2 times
    indexes = np.where(df_split['Sprungtyp'].value_counts() == min_num_jumps)       # search for the first time a jump with too few occurences comes up
    one_time_jumps = df_split['Sprungtyp'].value_counts()[indexes[0][0]:].index     # drop all jumps after that

    for jump in one_time_jumps:
        index = df_split[df_split['Sprungtyp'] == jump].index
        df_split.drop(index, inplace=True)

    # split train - test -> 80 : 20 -> stratify to keep distribution of classes
    df_train, df_test, y_train, y_test = train_test_split(df_split, df_split['Sprungtyp'], stratify=df_split['Sprungtyp'],
                                                          test_size=test_size, random_state=1)
    print("Split of Train/Test Data complete")

    # rewrite dataframes
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    column_names = list(df_test['Data'][0].columns)
    train_data = pd.DataFrame(columns=column_names)
    test_data = pd.DataFrame(columns=column_names)

    # unpack the dataframes to get the original structure back
    for i in range(len(df_train)):
        train_data = train_data.append(df_train.loc[i][1])
        print("Unpacking training Dataframe: " + str(i + 1) + "/" + str(len(df_train)))
    for j in range(len(df_test)):
        test_data = test_data.append(df_test.loc[j][1])
        print("Unpacking testing Dataframe: " + str(j + 1) + "/" + str(len(df_test)))

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    return train_data, test_data


def calc_avg_std(data):
    """
    Calculates the avg and std data for each column of each jump.

    :param data: data for calculation
    :return: avg_data dataframe and std_data dataframe
    """

    # rename all columns for unique identification
    avg_cols = list(data.columns)
    avg_ = ['avg_'] * (len(avg_cols) - 2)
    avg_cols.remove('Sprungtyp')
    avg_cols.remove('SprungID')
    avg_cols = list(map(str.__add__, avg_, list(avg_cols)))
    avg_cols.insert(0, 'Sprungtyp')
    avg_cols.insert(0, 'SprungID')

    std_cols = [value for value in list(data.columns) if value not in [col for col in data.columns if 'DJump' in col]]
    std_ = ['std_'] * (len(std_cols) - 2)
    std_cols.remove('Sprungtyp')
    std_cols.remove('SprungID')
    std_cols = list(map(str.__add__, std_, list(std_cols)))
    std_cols.insert(0, 'Sprungtyp')
    std_cols.insert(0, 'SprungID')

    averaged_data = pd.DataFrame(columns=avg_cols)

    std_data = pd.DataFrame(columns=std_cols)

    for id in data['SprungID'].unique():
        subframe = data[data['SprungID'] == id]

        mean = subframe.mean().to_frame().T
        std = subframe.std().to_frame().T

        avg_ = ['avg_'] * len(mean.columns)
        mean.columns = list(map(str.__add__, avg_, mean.columns))
        std_ = ['std_'] * len(std.columns)
        std.columns = list(map(str.__add__, std_, std.columns))

        mean['Sprungtyp'] = subframe['Sprungtyp'].unique()[0]
        mean['SprungID'] = subframe['SprungID'].unique()[0]
        std['Sprungtyp'] = subframe['Sprungtyp'].unique()[0]
        std['SprungID'] = subframe['SprungID'].unique()[0]

        averaged_data = averaged_data.append(mean, ignore_index=True)
        std_data = std_data.append(std, ignore_index=True)

        print(len(averaged_data) - 1)

    averaged_data = averaged_data.drop(columns=['avg_Time'])
    std_data = std_data.drop(columns=['std_Time'])

    # no preprocessed data in std_data as that would always be 0
    std_data = std_data.drop(columns=[col for col in std_data.columns if 'DJump' in col])

    return averaged_data, std_data


def combine_avg_std(avg_data, std_data):
    """
    Combines the avg_data and std_data in such a way that the average and std of a column are always beside each other

    :param avg_data: avg_data dataframe
    :param std_data: std_data dataframe
    :return: avg_std_data
    """

    avg_std_data = pd.merge(avg_data, std_data, on=['SprungID', 'Sprungtyp'], how='left')

    cols = [x for x in itertools.chain.from_iterable(itertools.zip_longest(list(avg_data.columns), list(std_data.columns))) if x]
    cols = list(dict.fromkeys(cols))

    avg_std_data = avg_std_data[cols]

    return avg_std_data


def percentage_cutting(data, percent_steps, method=None):
    """
    Creates data, which is separated by percent_steps.
    This is to reduce the number of datapoints per jump and depending on the method, use representative datapoints per percentage step.

    Relative to the percent_steps we get a different number of datapoints per jump. e.g. using 1% we get 100 datapoints, using 20% we get 5 datapoints

    :param data: Data to be turned into percentage based data
    :param percent_steps: value between 0 - 1, 100 has to be divisible by the value. We use [0.01, 0.02, 0.05, 0.1, 0.2, 0.25] -> [1%, 2%, 5%, 10%, 20%, 25%]
    :param method: None: Just keeps the original datapoints at each percentage step, 'mean': calculates the average value for each column in the percentage step, 'mean_std': calculates the average and std value for each column in the percentage step
    :return: Dataframe with percentage data
    """

    if method is None:
        df = pd.DataFrame(columns=data.columns)

        for jump in data['SprungID'].unique():
            print(jump)
            subframe = data[data['SprungID'] == jump]
            subframe.reset_index(drop=True, inplace=True)
            index_list = np.rint(np.arange(0, len(subframe), len(subframe) * percent_steps))
            if len(index_list) != 100 / (percent_steps * 100):
                index_list = index_list[:-1]
            df = df.append(subframe.iloc[index_list], ignore_index=True)
        df = df.drop(['Time'], axis=1)

    if method == 'mean':
        df = pd.DataFrame(columns=data.columns)

        for jump in data['SprungID'].unique():
            print(jump)
            subframe = data[data['SprungID'] == jump]
            subframe.reset_index(drop=True, inplace=True)
            index_list = np.rint(np.arange(0, len(subframe), len(subframe) * percent_steps))
            if index_list[-1] != len(subframe):
                index_list = np.append(index_list, len(subframe))
            id = subframe['SprungID'][0]
            jump_type = subframe['Sprungtyp'][0]
            for i in range(len(index_list) - 1):
                start = int(index_list[i])
                end = int(index_list[i + 1])
                temp = subframe.iloc[start:end].mean()
                temp = temp.to_frame().transpose()
                temp = temp.drop(['Time'], axis=1)
                temp.insert(0, 'SprungID', id)
                temp.insert(1, 'Sprungtyp', jump_type)

                df = df.append(temp, ignore_index=True)
        df = df.drop(['Time'], axis=1)

    if method == 'mean_std':
        cols = [value for value in list(data.columns) if value not in [col for col in data.columns if 'DJump' in col]]
        cols.remove('Sprungtyp')
        cols.remove('SprungID')
        cols.remove('Time')
        mean_ = ['mean_'] * len(cols)
        mean_cols = list(map(str.__add__, mean_, cols))
        mean_cols.insert(0, 'Sprungtyp')
        mean_cols.insert(0, 'SprungID')

        std_ = ['std_'] * len(cols)
        std_cols = list(map(str.__add__, std_, cols))
        std_cols.insert(0, 'Sprungtyp')
        std_cols.insert(0, 'SprungID')

        cols = [x for x in itertools.chain.from_iterable(itertools.zip_longest(mean_cols, std_cols)) if x]
        cols = list(dict.fromkeys(cols))
        cols.extend([col for col in data.columns if 'DJump' in col])

        df = pd.DataFrame(columns=cols)

        for jump in data['SprungID'].unique():
            print(jump)
            subframe = data[data['SprungID'] == jump]
            subframe.reset_index(drop=True, inplace=True)
            index_list = np.rint(np.arange(0, len(subframe), len(subframe) * percent_steps))
            if index_list[-1] != len(subframe):
                index_list = np.append(index_list, len(subframe))
            id = subframe['SprungID'][0]
            jump_type = subframe['Sprungtyp'][0]
            for i in range(len(index_list) - 1):
                start = int(index_list[i])
                end = int(index_list[i + 1])

                mean = subframe.iloc[start:end].mean()
                std = subframe.iloc[start:end].std()
                mean = mean.to_frame().transpose()
                std = std.to_frame().transpose()
                mean = mean.drop(['Time'], axis=1)
                std = std.drop(['Time'], axis=1)
                std = std.drop([col for col in std.columns if 'DJump' in col], axis=1)

                cols = [value for value in list(mean.columns) if value not in [col for col in mean.columns if 'DJump' in col]]
                mean_ = ['mean_'] * len(cols)
                mean_cols = list(map(str.__add__, mean_, cols))
                mean_cols.extend([col for col in mean.columns if 'DJump' in col])

                cols = [value for value in list(std.columns) if value not in [col for col in std.columns if 'DJump' in col]]
                std_ = ['std_'] * len(cols)
                std_cols = list(map(str.__add__, std_, cols))

                mean.columns = mean_cols
                std.columns = std_cols

                temp = pd.concat([mean, std], axis=1)
                temp.insert(0, 'SprungID', id)
                temp.insert(1, 'Sprungtyp', jump_type)

                df = df.append(temp, ignore_index=True)
    return df


def vectorize(data):
    """
    vectorizes data by putting the datapoints of each jump into one row.

    :param data: data to vectorize
    :return: vectorized data
    """

    first = True
    for jump in data['SprungID'].unique():
        print(jump)
        subframe = data[data['SprungID'] == jump]
        subframe.reset_index(drop=True, inplace=True)
        equal_data_names = ['SprungID', 'Sprungtyp']
        equal_data_names.extend([col for col in subframe.columns if 'DJump' in col])
        equal_data = subframe[equal_data_names].iloc[0]
        equal_data = equal_data.to_frame().transpose()
        percentage_list = np.rint(np.arange(0, 100, 100 / len(subframe)))
        # new
        for row in range(len(subframe)):
            name = str(int(percentage_list[row]))
            diff_names = [value for value in list(data.columns) if value not in equal_data_names]
            copy = subframe[diff_names].iloc[
                int(row)]  # data.columns - equal_data.columns --> to get not equal data column names for every dataset
            copy = copy.to_frame().transpose()
            # percantage 'name' combined with diff_names
            percentage_name = [name + '_'] * len(copy.columns)
            full_name = list(map(str.__add__, percentage_name, diff_names))
            copy.columns = full_name

            copy.reset_index(drop=True, inplace=True)
            equal_data = pd.concat([equal_data, copy], axis=1)

        if first is True:
            df = pd.DataFrame(columns=equal_data.columns)
            first = False
        df = df.append(equal_data, ignore_index=True)

    return df


def mean_jump_generator(data):
    """
    Create one average jump for each type of jump.

    :param data: data to be used
    :return: dataframe with all average jumps
    """

    # only for vectorized data
    unique_jumps = data['Sprungtyp'].unique()
    df = pd.DataFrame()
    for jump in unique_jumps:
        subframe = data[data['Sprungtyp'] == jump]
        if len(subframe) > 2:
            mean_jump = subframe.mean().to_frame().T
            mean_jump['Sprungtyp'] = jump
            mean_jump['SprungID'] = 'generated_Mean_' + jump
            df = df.append(mean_jump, ignore_index=True)
    return df


def devectorize(data):
    """
    Devectorizes a dataframe to use with CNNs.

    :param data: data to be devectorized
    :return: devectorized dataframe
    """

    percent = 100 / ((len([col for col in data.columns if 'DJump' not in col]) - 2) / len([col for col in data.columns if col.startswith('0')]))
    percent_list = np.rint(np.arange(0, 100, percent))
    names = [col for col in data.columns if col.startswith(str(int(percent_list[0])))]
    without_percent_name = []
    for name in names:
        without_percent_name.append(name.replace('0_', ''))
    namelist = ['SprungID', 'Sprungtyp']
    namelist.extend(without_percent_name)
    namelist.extend([col for col in data.columns if 'DJump' in col])
    df = pd.DataFrame(columns=namelist)
    for row in data.iterrows():
        temp = pd.DataFrame()
        row = row[1]    # get the series
        row = row.to_frame().T.reset_index(drop=True)
        for p in percent_list:
            n = row[[col for col in row.columns if col.startswith(str(int(p))+'_')]]
            n.columns = without_percent_name
            temp = temp.append(n, ignore_index=True)
        temp['Sprungtyp'] = row['Sprungtyp'][0]
        temp['SprungID'] = row['SprungID'][0]
        djumps = row[[col for col in row.columns if 'DJump' in col]]
        djumps['SprungID'] = row['SprungID'][0]
        temp = pd.merge(temp, djumps, on='SprungID')

        df = df.append(temp, ignore_index=True)
    return df


"""
#                                       #
#                                       #
#           Main Functions              #
#                                       #
#                                       #
"""


def make_all_data():
    """
    Use all Rohdaten and Sprungzuordnung to create one big dataset with all data except for pauses.

    We split the creation of the big all_data into a couple smaller fractions to save on memory.
    After that, we can combine all the fractions.

    The fractions get saved in Sprungdaten_processed/all_data/...
    and all_data.csv gets saved in Sprungdaten_processed/all_data.csv

    :return:
    """

    first_round = True

    folders = os.listdir('Sprungdaten Innotramp')
    # folders = ['2019.09.30']              # for running only 1 folder
    for folder in folders:
        files = os.listdir("Sprungdaten Innotramp/" + folder)
        csv_files = [f for f in files if f.endswith('.csv')]
        xlsx_files = [f for f in files if f.endswith('.XLSX')]

        for i in range(len(csv_files)):
            csv_data = read_raw_csv_data("Sprungdaten Innotramp/" + folder + "/" + csv_files[i])
            xlsx_data = read_xlsx_data("Sprungdaten Innotramp/" + folder + "/" + xlsx_files[i])

            csv_data['Time'] = np.round(csv_data['Time'].apply(convert_comma_to_dot), 3)  # convert time stucture
            xlsx_data["Zeit"] = xlsx_data["Zeit"].apply(convert_to_numeric)

            temp_data = pd.DataFrame(columns=csv_data.drop(['Dist'], axis=1).columns)  # df for csv data
            temp_data = temp_data.append(csv_data.drop(['Dist'], axis=1))

            cumultime = np.cumsum(xlsx_data['Zeit'].to_numpy(), dtype=float)  # cumultative sum to make the time equal to the rohdaten
            xlsx_data['cumultime'] = cumultime

            start_time = csv_data['Time'][0]

            global_xlsx_col_names = ['SprungID', 'Sprungtyp'] + [col for col in xlsx_data.columns if 'DJump' in col]
            sprungzuordnung = pd.DataFrame(columns=global_xlsx_col_names + ['Time'])  # create df with needed column names

            for row in xlsx_data.iterrows():

                end_time = row[1]['cumultime']
                join_times = np.arange(start_time, end_time, 0.002)  # creates all times with 0.002 steps from start to end
                sprungID = row[1]['Messung'] + "-" + str(row[1]['Lap#'])  # create an unique ID for each jump

                multiply_array = np.array(
                    [sprungID, row[1]['Sprungtyp']])  # create xlsx array, with only columns that needed
                for col in xlsx_data.columns:
                    if 'DJump' in col:
                        multiply_array = np.append(multiply_array, row[1][col])
                multiply_array = np.transpose(np.repeat(multiply_array.reshape(len(multiply_array), 1), len(join_times),
                                                        axis=1))  # this array will be multiplied when joined with the rohdaten

                temp_sprungzuordnung = pd.DataFrame(multiply_array, columns=global_xlsx_col_names)
                temp_sprungzuordnung['Time'] = np.round(join_times, 3)  # add time column for merging with Rohdaten

                sprungzuordnung = sprungzuordnung.append(temp_sprungzuordnung)

                start_time = end_time

            temp_data = pd.merge(temp_data, sprungzuordnung, on='Time', how='left')
            temp_data.drop(temp_data[temp_data['Sprungtyp'] == 'nan'].index, inplace=True)  # remove breaks to narrow the dataset
            if first_round:
                all_data = pd.DataFrame(columns=temp_data.columns)
                first_round = False
            all_data = all_data.append(temp_data, ignore_index=True)

        for col in all_data.columns:
            if col in ['Acc_N_Fil', 'Gyro_x_R', 'Gyro_y_R', 'Gyro_z_R', 'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil']:
                all_data[col] = all_data[col].apply(convert_comma_to_dot)

        if not os.path.isdir("all_data"):
            os.makedirs("Sprungdaten_processed/all_data")
        all_data.to_csv('Sprungdaten_processed/all_data/all_data_' + folder + '.csv', index=False)  # save smaller datasets with name of the folder
        all_data = all_data[0:0]    # clear dataframe

    print("Finished creating all all_data fractions")

    files = [data for data in os.listdir('Sprungdaten_processed/all_data') if 'all_data' in data]
    col_names = pd.read_csv('Sprungdaten_processed/all_data/' + files[0]).columns
    all_data = pd.DataFrame(columns=col_names)
    for file in files:
        data = pd.read_csv('Sprungdaten_processed/all_data/' + file)
        data.columns = col_names
        all_data = all_data.append(data, ignore_index=True)
    all_data.to_csv("Sprungdaten_processed/all_data.csv", index=False)

    print("Finished combining all_data")


def make_data_only_jumps(num_strecksprung=300):
    """
    Creates data, where things like Datenfehler and Einturnen are non existent and we have the minimum amount of classes
    through spelling correction.

    We save a copy with preprocessed data and one without preprocessed data

    :param num_strecksprung: The number of Strecksprünge to which it should undersample. default=300
    """

    all_data = pd.read_csv("Sprungdaten_processed/all_data.csv")
    data_only_jumps = sort_out_errors(all_data)

    # undersample strecksprung
    subframe = data_only_jumps[data_only_jumps['Sprungtyp'] == 'Strecksprung']
    ids = subframe['SprungID'].unique()
    chosen_ids = np.random.choice(ids, num_strecksprung, replace=False)

    ids_to_delete = list(set(ids) - set(chosen_ids))
    data_only_jumps = data_only_jumps[~data_only_jumps['SprungID'].isin(ids_to_delete)]

    save_as_csv(data_only_jumps, "data_only_jumps", folder="with_preprocessed")
    print("Saved data_only_jumps with preprocessed")

    data_only_jumps_without_pp = data_only_jumps.drop([col for col in data_only_jumps.columns if 'DJump' in col], axis=1)
    save_as_csv(data_only_jumps_without_pp, "data_only_jumps", folder="without_preprocessed")
    print("Saved data_only_jumps without preprocessed")


def make_avg_std_data(preprocessed):
    """
    Creates 3 datasets.

    avg_data: The average of each column for each jump

    std_data: The standard deviation of each column for each jump

    avg_std_data: avg_data and std_data combined with the corresponding columns besides each other

    :param preprocessed: 'with_preprocessed' or 'without_preprocessed'
    """

    if preprocessed not in ['with_preprocessed', 'without_preprocessed']:
        raise AttributeError("Variable preprocessed not correctly defined. Try 'with_preprocessed' or 'without_preprocessed'")

    data_only_jumps = pd.read_csv('Sprungdaten_processed/' + preprocessed + "/data_only_jumps.csv")

    averaged_data, std_data = calc_avg_std(data_only_jumps)
    save_as_csv(averaged_data, "averaged_data", folder=preprocessed + "/averaged_data")
    save_as_csv(std_data, "std_data", folder=preprocessed + "/std_data")

    avg_std_data = combine_avg_std(averaged_data, std_data)
    save_as_csv(avg_std_data, "avg_std_data", folder=preprocessed + "/avg_std_data")

    # make train/test splits
    train_data, test_data = split_train_test(averaged_data)
    save_as_csv(train_data, 'averaged_data_train', folder=preprocessed + '/averaged_data')
    save_as_csv(test_data, 'averaged_data_test', folder=preprocessed + '/averaged_data')

    train_data, test_data = split_train_test(std_data)
    save_as_csv(train_data, 'std_data_train', folder=preprocessed + '/std_data')
    save_as_csv(test_data, 'std_data_test', folder=preprocessed + '/std_data')

    train_data, test_data = split_train_test(avg_std_data)
    save_as_csv(train_data, 'avg_std_data_train', folder=preprocessed + '/avg_std_data')
    save_as_csv(test_data, 'avg_std_data_test', folder=preprocessed + '/avg_std_data')


def make_percentage_data(preprocessed, percentage_steps, method):
    """
    Creates all the percentage data specified in the parameters.

    :param preprocessed: 'with_preprocessed' or 'without_preprocessed'
    :param percentage_steps: a list with fraction values which divide 100. We use ['0.25', '0.20', '0.10', '0.05', '0.02', '0.01']
    :param method: None, 'mean', or 'mean_std'
    """

    if preprocessed not in ['with_preprocessed', 'without_preprocessed']:
        raise AttributeError("Variable preprocessed not correctly defined. Try 'with_preprocessed' or 'without_preprocessed'")

    if type(percentage_steps) != list:
        raise AttributeError("Variable percentage_steps not correctly defined. It needs to be a list!")
    for p in percentage_steps:
        assert (100 / p) % 2 == 0.0, "The percentage_step should be able to divide 100"

    if method not in [None, 'mean', 'mean_std']:
        raise AttributeError("Variable method not correctly defined. Try None, 'mean', or 'mean_std'")

    data_only_jumps = pd.read_csv('Sprungdaten_processed/' + preprocessed + "/data_only_jumps.csv")

    for percent in percentage_steps:
        if method is None:
            data = percentage_cutting(data_only_jumps, percent)
            save_as_csv(data, 'percentage_' + str(int(percent * 100)),
                        folder=preprocessed + '/percentage/' + str(int(percent * 100)))

            train_data, test_data = split_train_test(data)

            save_as_csv(train_data, 'percentage_' + str(int(percent * 100)) + '_train',
                        folder=preprocessed + '/percentage/' + str(int(percent * 100)))
            save_as_csv(test_data, 'percentage_' + str(int(percent * 100)) + '_test',
                        folder=preprocessed + '/percentage/' + str(int(percent * 100)))
        else:
            data = percentage_cutting(data_only_jumps, percent, method)
            save_as_csv(data, 'percentage_' + method + '_' + str(int(percent * 100)),
                        folder=preprocessed + '/percentage/' + str(int(percent * 100)))

            train_data, test_data = split_train_test(data)

            save_as_csv(train_data, 'percentage_' + method + '_' + str(int(percent * 100)) + '_train',
                        folder=preprocessed + '/percentage/' + str(int(percent * 100)))
            save_as_csv(test_data, 'percentage_' + method + '_' + str(int(percent * 100)) + '_test',
                        folder=preprocessed + '/percentage/' + str(int(percent * 100)))


def vectorize_data(preprocessed, name, percentages):
    """
    Vectorizes the percentage data previously created.

    :param preprocessed: 'with_preprocessed' or 'without_preprocessed'
    :param name: 'percentage_mean_std', 'percentage_mean', or 'percentage'
    :param percentages: a list with values which divide 100. We use ['25', '20', '10', '5', '2', '1']
    """

    if preprocessed not in ['with_preprocessed', 'without_preprocessed']:
        raise AttributeError("Variable preprocessed not correctly defined. Try 'with_preprocessed' or 'without_preprocessed'")

    if name not in ['percentage_mean_std', 'percentage_mean', 'percentage']:
        raise AttributeError("Variable method not correctly defined. Try None, 'mean', or 'mean_std'")

    if type(percentages) != list:
        raise AttributeError("Variable percentages not correctly defined. It needs to be a list!")
    for p in percentages:
        assert (100 / int(p)) % 2 == 0.0, "The percentages should be able to divide 100. Try ['25', '20', '10', '5', '2', '1']"

    name = name + '_'
    for percent in percentages:  # '25', '20', '10', '5', '2', '1'
        data = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/' + percent + '/' + name + percent + '.csv')
        train_data = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/' + percent + '/' + name + percent + '_train' + '.csv')
        test_data = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/' + percent + '/' + name + percent + '_test' + '.csv')

        vector_data = vectorize(data)

        vector_train_data = vectorize(train_data)
        vector_test_data = vectorize(test_data)

        save_as_csv(vector_train_data, 'vector_' + name + percent + '_train', folder=preprocessed + '/percentage/' + percent)
        save_as_csv(vector_test_data, 'vector_' + name + percent + '_test', folder=preprocessed + '/percentage/' + percent)
        save_as_csv(vector_data, 'vector_' + name + percent, folder=preprocessed + '/percentage/' + percent)


def make_AJ_jumps(preprocessed, name, percentages):
    """
    Creates datasets with all jumps of a type averaged into one representative.

    It automatically generates a vectorized and unvectorized version.

    :param preprocessed: 'with_preprocessed' or 'without_preprocessed'
    :param name: 'percentage_mean_std', 'percentage_mean', or 'percentage'
    :param percentages: a list with values which divide 100. We use ['25', '20', '10', '5', '2', '1']
    """

    if preprocessed not in ['with_preprocessed', 'without_preprocessed']:
        raise AttributeError("Variable preprocessed not correctly defined. Try 'with_preprocessed' or 'without_preprocessed'")

    if name not in ['percentage_mean_std', 'percentage_mean', 'percentage']:
        raise AttributeError("Variable method not correctly defined. Try None, 'mean', or 'mean_std'")

    if type(percentages) != list:
        raise AttributeError("Variable percentages not correctly defined. It needs to be a list!")
    for p in percentages:
        assert (100 / int(p)) % 2 == 0.0, "The percentages should be able to divide 100. Try ['25', '20', '10', '5', '2', '1']"

    name = name + '_'
    for p in percentages:  # ['25', '20', '10', '5', '2', '1']
        print(p)
        data = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/' + p + '/vector_' + name + p + '.csv')
        df = mean_jump_generator(data)
        save_as_csv(df, 'vector_AJ_' + name + p, folder=preprocessed + '/percentage/' + p)
        devec_df = devectorize(df)
        save_as_csv(devec_df, 'AJ_' + name + p, folder=preprocessed + '/percentage/' + p)


def main():
    make_AJ_jumps('without_preprocessed', 'percentage', ['25'])


if __name__ == '__main__':
    main()
