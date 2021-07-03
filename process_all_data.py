import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import itertools


def read_data(name):
    all_data = pd.read_csv("Sprungdaten_processed/" + name + "/" + name + ".csv")

    return all_data


def sort_out_errors(all_data):
    # delete all the unwanted data
    all_data = all_data.query(
        'Sprungtyp != "Datenfehler" & Sprungtyp != "Fehlerhafte Daten" & Sprungtyp != "Unbekannt" & Sprungtyp != "Einturnen"')

    # delete the nan values
    nan_value = float('NaN')
    all_data['Sprungtyp'].replace("", nan_value, inplace=True)
    all_data.dropna(subset=['Sprungtyp'], inplace=True)

    # delete jumps where DJumps all 0
    djumps = [col for col in all_data.columns if 'DJump' in col]
    # zero = pd.DataFrame(np.zeros((1, len(djumps))), columns=djumps)

    all_data_djumps = all_data[djumps]
    all_data = all_data[(all_data_djumps == 0).sum(1) < len(djumps)]
    all_data = all_data.reset_index(drop=True)

    return all_data


def save_as_csv(data, name, folder=""):
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
        data.to_csv('Sprungdaten_processed/' + folder + "/" + name + ".csv", index=False)


def delete_0_point_jumps(data):
    data = data.query(
        'Sprungtyp != "Strecksprung" & Sprungtyp != "Standsprung" & Sprungtyp != "Hocksprung" & Sprungtyp != "Bücksprung" '
        '& Sprungtyp != "Grätschwinkel" & Sprungtyp != "Sitzsprung" & Sprungtyp != "00V"')

    return data


def correct_space_errors(data):
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


def only_marked_jumps(data):
    # TODO

    data = data.query(
        'Sprungtyp == "Fliffis aus C" or Sprungtyp == "Fliffis aus B" or Sprungtyp == "Fliffis- Rudi C" or Sprungtyp == "Fliffis- Rudi B"'
        'or Sprungtyp == "Voll- ein- halb- aus C" or Sprungtyp == "Voll- ein- halb- aus B" or Sprungtyp == "Voll- ein- Rudi- aus A"'
        'or Sprungtyp == "Triffis C" or Sprungtyp == "Triffis B" or Sprungtyp == "Triffis- Rudi B" or Sprungtyp == "Schraubensalto"'
        'or Sprungtyp == "Doppelsschraubensalto" or Sprungtyp == "Doppelsalto C" or Sprungtyp == "Doppelsalto B" or Sprungtyp == "Doppelsalto A"'
        'or Sprungtyp == "1/2 ein 1/2 aus C" or Sprungtyp == "1/2 ein 1/2 aus B" or Sprungtyp == "Halb- ein- Rudy- aus B" or Sprungtyp == "Halb- ein- Rudy- aus C"'
        'or Sprungtyp == "Voll- ein- voll- aus C" or Sprungtyp == "Voll- ein- voll- aus A" or Sprungtyp == "Voll- ein- doppel- aus A"'
        'or Sprungtyp == "1/2 ein 1/2 aus Triffis C" or Sprungtyp == "1/2 ein 1/2 aus Triffis B"')

    print(data['Sprungtyp'].unique())
    return data


def convert_comma_to_dot(x):
    return float(x.replace(',', '.'))


def calc_avg(data):

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

    std_data = std_data.drop(columns=[col for col in std_data.columns if 'DJump' in col])

    print(averaged_data)
    print(std_data)
    return averaged_data, std_data


def class_std_mean(avg, std):
    avg_std_data = pd.DataFrame(columns=['Sprungtyp', 'avg_ACC_N', 'avg_ACC_N_ROT_filtered',
                                         'avg_Acc_x_Fil', 'avg_Acc_y_Fil', 'avg_Acc_z_Fil', 'avg_Gyro_x_Fil',
                                         'avg_Gyro_y_Fil', 'avg_Gyro_z_Fil',
                                         'avg_DJump_SIG_I_x LapEnd', 'avg_DJump_SIG_I_y LapEnd',
                                         'avg_DJump_SIG_I_z LapEnd',
                                         'avg_DJump_Abs_I_x LapEnd', 'avg_DJump_Abs_I_y LapEnd',
                                         'avg_DJump_Abs_I_z LapEnd',
                                         'std_ACC_N', 'std_ACC_N_ROT_filtered',
                                         'std_Acc_x_Fil', 'std_Acc_y_Fil', 'std_Acc_z_Fil', 'std_Gyro_x_Fil',
                                         'std_Gyro_y_Fil', 'std_Gyro_z_Fil'])

    for typ in avg['Sprungtyp'].unique():
        avg_subframe = avg[avg['Sprungtyp'] == typ]
        std_subframe = std[std['Sprungtyp'] == typ]

        avg_mean = avg_subframe.mean()
        std_mean = std_subframe.mean()

        avg_mean = avg_mean.to_frame().transpose()
        std_mean = std_mean.to_frame().transpose()
        avg_mean.columns = ['avg_ACC_N', 'avg_ACC_N_ROT_filtered',
                            'avg_Acc_x_Fil', 'avg_Acc_y_Fil', 'avg_Acc_z_Fil', 'avg_Gyro_x_Fil', 'avg_Gyro_y_Fil',
                            'avg_Gyro_z_Fil',
                            'avg_DJump_SIG_I_x LapEnd', 'avg_DJump_SIG_I_y LapEnd', 'avg_DJump_SIG_I_z LapEnd',
                            'avg_DJump_Abs_I_x LapEnd', 'avg_DJump_Abs_I_y LapEnd', 'avg_DJump_Abs_I_z LapEnd']
        std_mean.columns = ['std_ACC_N', 'std_ACC_N_ROT_filtered', 'std_Acc_x_Fil', 'std_Acc_y_Fil', 'std_Acc_z_Fil',
                            'std_Gyro_x_Fil', 'std_Gyro_y_Fil', 'std_Gyro_z_Fil']

        result = pd.concat([avg_mean, std_mean], axis=1)
        result.insert(0, "Sprungtyp", typ)

        avg_std_data = avg_std_data.append(result)

    avg_std_data.to_excel(os.getcwd() + "/Sprungdaten_processed/" + "class_std_mean.xlsx")

    return


def normalize(data):
    for (column_name, column_data) in data.iteritems(): # TODO

        if column_name in ['ACC_N', 'ACC_N_ROT_filtered', 'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil',
                           'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil',
                           'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                           'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd']:
            max_value = data[column_name].max()
            min_value = data[column_name].min()

            data[column_name] = (data[column_name] - min_value) / (max_value - min_value)

    return data


def split_train_test(data):
    # if the data consists of only one row per jump we can use the easy way
    if len(data['SprungID'].unique()) == len(data):
        # drop the jumps which occur <= 2 times
        indexes = np.where(data['Sprungtyp'].value_counts() == 2)
        one_time_jumps = data['Sprungtyp'].value_counts()[indexes[0][0]:].index

        for jump in one_time_jumps:
            index = data[data['Sprungtyp'] == jump].index
            data.drop(index, inplace=True)

        # split train - test -> 80 : 20 -> stratify to keep distribution of classes
        df_train, df_test, y_train, y_test = train_test_split(data, data['Sprungtyp'], stratify=data['Sprungtyp'],
                                                              test_size=0.2, random_state=1)
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
    indexes = np.where(df_split['Sprungtyp'].value_counts() == 2)
    one_time_jumps = df_split['Sprungtyp'].value_counts()[indexes[0][0]:].index

    for jump in one_time_jumps:
        index = df_split[df_split['Sprungtyp'] == jump].index
        df_split.drop(index, inplace=True)

    # split train - test -> 80 : 20 -> stratify to keep distribution of classes
    df_train, df_test, y_train, y_test = train_test_split(df_split, df_split['Sprungtyp'],
                                                          stratify=df_split['Sprungtyp'], test_size=0.2, random_state=1)
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


def combine_avg_std(avg_data, std_data):

    avg_std_data = pd.merge(avg_data, std_data, on=['SprungID', 'Sprungtyp'], how='left')

    cols = [x for x in itertools.chain.from_iterable(itertools.zip_longest(list(avg_data.columns), list(std_data.columns))) if x]
    cols = list(dict.fromkeys(cols))

    avg_std_data = avg_std_data[cols]

    return avg_std_data


def make_jump_length_consistent(data, method="cut_last"):
    ids = data['SprungID'].value_counts()
    min_length = ids[-1]

    df = pd.DataFrame(columns=['SprungID', 'Sprungtyp', 'Time', 'TimeInJump', 'ACC_N', 'ACC_N_ROT_filtered',
                               'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil', 'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil',
                               'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                               'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])
    if method == "cut_last":

        for id in data['SprungID'].unique():
            subframe = data[data['SprungID'] == id]
            subframe.reset_index(drop=True, inplace=True)
            df = df.append(subframe[:min_length], ignore_index=True)

    if method == "cut_first":

        for id in data['SprungID'].unique():
            subframe = data[data['SprungID'] == id]
            subframe.reset_index(drop=True, inplace=True)
            df = df.append(subframe[len(subframe) - min_length:], ignore_index=True)

    if method == "padding_0":

        max_length = ids[0]

        for id in data['SprungID'].unique():
            subframe = data[data['SprungID'] == id]
            subframe.reset_index(drop=True, inplace=True)
            if len(subframe) < max_length:
                length_to_fill = max_length - len(subframe)
                zero_list = [0] * length_to_fill
                time_list = np.arange(round(subframe['Time'].iloc[-1] + 0.002, 3),
                                      round(subframe['Time'].iloc[-1] + (0.002 * length_to_fill + 0.001), 3), 0.002)
                timeinjump_list = np.arange(round(subframe['TimeInJump'].iloc[-1] + 0.002, 3),
                                            round(subframe['TimeInJump'].iloc[-1] + (0.002 * length_to_fill + 0.001),
                                                  3), 0.002)
                jump_ids = [subframe['SprungID'].iloc[-1]] * length_to_fill
                jump_typ = [subframe['Sprungtyp'].iloc[-1]] * length_to_fill

                """
                temp_df = pd.DataFrame([jump_ids, jump_typ, time_list, timeinjump_list, zero_list, zero_list, zero_list,
                                        zero_list, zero_list, zero_list, zero_list, zero_list, zero_list, zero_list, zero_list, zero_list, zero_list, zero_list],
                                        columns=['SprungID', 'Sprungtyp', 'Time', 'TimeInJump', 'ACC_N', 'ACC_N_ROT_filtered',
                                                 'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil', 'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil',
                                                 'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                                 'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])
                                                 """
                temp_df = pd.DataFrame(
                    data={'SprungID': jump_ids, 'Sprungtyp': jump_typ, 'Time': time_list, 'TimeInJump': timeinjump_list,
                          'ACC_N': zero_list, 'ACC_N_ROT_filtered': zero_list,
                          'Acc_x_Fil': zero_list, 'Acc_y_Fil': zero_list, 'Acc_z_Fil': zero_list,
                          'Gyro_x_Fil': zero_list, 'Gyro_y_Fil': zero_list, 'Gyro_z_Fil': zero_list,
                          'DJump_SIG_I_x LapEnd': zero_list, 'DJump_SIG_I_y LapEnd': zero_list,
                          'DJump_SIG_I_z LapEnd': zero_list,
                          'DJump_Abs_I_x LapEnd': zero_list, 'DJump_Abs_I_y LapEnd': zero_list,
                          'DJump_Abs_I_z LapEnd': zero_list})

                df = df.append(subframe, ignore_index=True)
                df = df.append(temp_df, ignore_index=True)

    return df


def percentage_cutting(data, percent_steps, method=None):
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


def create_shap_samples(data):
    unique_jumps = data['Sprungtyp'].unique()
    df = pd.DataFrame()
    for jump in unique_jumps:
        subframe = data[data['Sprungtyp'] == jump]
        ids = subframe['SprungID'].unique()
        if len(ids) > 2:
            random = np.random.choice(ids, size=3, replace=False)
            df = df.append(data[data['SprungID'].isin(random)], ignore_index=True)
    return df


def mean_jump_generator(data):
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


def main():

    """
    all_data = read_data("all_data")
    data_only_jumps = sort_out_errors(all_data)

    count_jumptypes = data_only_jumps['Sprungtyp'].value_counts()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(count_jumptypes)

    save_as_csv(data_only_jumps, 'data_only_jumps.csv')
    """

    """
    data_only_jumps = read_data("data_only_jumps")
    data_point_jumps = delete_0_point_jumps(data_only_jumps)
    save_as_csv(data_point_jumps, "data_point_jumps.csv")
    """

    """
    data_point_jumps = read_data("data_point_jumps")
    data_point_jumps = correct_space_errors(data_point_jumps)
    print(data_point_jumps['Sprungtyp'].unique())
    save_as_csv(data_point_jumps, "data_point_jumps.csv")
    """

    """
    data_point_jumps = read_data("data_point_jumps")
    marked_jumps = only_marked_jumps(data_point_jumps)
    """

    """
    data_point_jumps = read_data("data_point_jumps")
    averaged_data, std_data = calc_avg(data_point_jumps)
    save_as_csv(averaged_data, "averaged_data.csv", with_time=False)
    save_as_csv(std_data, "std_data.csv", with_time=False)
    """

    """
    averaged_data = read_data("averaged_data")
    std_data = read_data("std_data.csv")
    class_std_mean(averaged_data, std_data)
    """

    """
    data_point_jumps = read_data("data_point_jumps")
    normalized_data = normalize(data_point_jumps)
    save_as_csv(normalized_data, "normalized_data.csv")
    """

    """
    for file in ['averaged_data']:
        data = read_data(file)
        #data = pd.read_csv("Sprungdaten_processed/" + "same_length" + "/" + "same_length_" + file + ".csv")
        train_data, test_data = split_train_test(data)
        save_as_csv(train_data, file + "_train", folder=file)
        save_as_csv(test_data, file + "_test", folder=file)
        #save_as_csv(train_data, "same_length_" + file + "_train", folder="same_length")
        #save_as_csv(test_data, "same_length_" + file + "_test", folder="same_length")
    """

    # for generating avg data
    """
    averaged_data = read_data("averaged_data")
    std_data = read_data("std_data")
    avg_std_data = combine_avg_std(averaged_data, std_data)
    save_as_csv(avg_std_data, "avg_std_data", folder="avg_std_data")
    """

    # for Cutting jumps on same length
    """
    for method in ['cut_first', 'cut_last', 'padding_0']:
        data_point_jumps = read_data("data_point_jumps")
        data = make_jump_length_consistent(data_point_jumps, method=method)
        save_as_csv(data, "same_length_" + method, folder="same_length")
    """

    # create data_only_jumps
    """
    files = [data for data in os.listdir('Sprungdaten_processed') if 'all_data' in data]
    col_names = pd.read_csv('Sprungdaten_processed/' + files[0]).columns
    all_data = pd.DataFrame(columns=col_names)
    for file in files:
        data = pd.read_csv('Sprungdaten_processed/' + file)
        data.columns = col_names
        all_data = all_data.append(data, ignore_index=True)
    all_data.to_csv("Sprungdaten_processed/all_data.csv", index=False)
    """

    # Sorting Out Errors and undersampling Strecksprünge
    """
    all_data = pd.read_csv("Sprungdaten_processed/all_data.csv")
    data_only_jumps = sort_out_errors(all_data)
    data_only_jumps = correct_space_errors(data_only_jumps)

    # undersample strecksprung
    subframe = data_only_jumps[data_only_jumps['Sprungtyp'] == 'Strecksprung']
    ids = subframe['SprungID'].unique()
    chosen_ids = np.random.choice(ids, 300, replace=False)

    ids_to_delete = list(set(ids) - set(chosen_ids))
    data_only_jumps = data_only_jumps[~data_only_jumps['SprungID'].isin(ids_to_delete)]

    save_as_csv(data_only_jumps, "data_only_jumps", folder="with_preprocessed")
    """

    # ___________________________________________________
    # with preprocessed data or without preprocessed data
    pp = 'with_preprocessed'
    data_only_jumps = pd.read_csv("Sprungdaten_processed/" + pp + "/data_only_jumps.csv")

    """
    averaged_data, std_data = calc_avg(data_only_jumps)
    save_as_csv(averaged_data, "averaged_data", folder=pp + "/averaged_data")
    save_as_csv(std_data, "std_data", folder=pp + "/std_data")

    avg_std_data = combine_avg_std(averaged_data, std_data)
    save_as_csv(avg_std_data, "avg_std_data", folder=pp + "/avg_std_data")
    
    averaged_data = pd.read_csv('Sprungdaten_processed/' + pp + '/averaged_data/averaged_data.csv')
    std_data = pd.read_csv('Sprungdaten_processed/' + pp + '/std_data/std_data.csv')
    avg_std_data = pd.read_csv('Sprungdaten_processed/' + pp + '/avg_std_data/avg_std_data.csv')
    
    train_data, test_data = split_train_test(averaged_data)
    save_as_csv(train_data, 'averaged_data_train', folder=pp + '/averaged_data')
    save_as_csv(test_data, 'averaged_data_test', folder=pp + '/averaged_data')

    train_data, test_data = split_train_test(std_data)
    save_as_csv(train_data, 'std_data_train', folder=pp + '/std_data')
    save_as_csv(test_data, 'std_data_test', folder=pp + '/std_data')

    train_data, test_data = split_train_test(avg_std_data)
    save_as_csv(train_data, 'avg_std_data_train', folder=pp + '/avg_std_data')
    save_as_csv(test_data, 'avg_std_data_test', folder=pp + '/avg_std_data')
    """

    # for percentage
    #"""
    param = 'mean'  # 'mean' or None
    for percent in [0.01]:
        if param is None:
            data = percentage_cutting(data_only_jumps, percent)
            save_as_csv(data, 'percentage_'+ str(int(percent * 100)), 
                        folder=pp + '/percentage/' + str(int(percent * 100)))
    
            train_data, test_data = split_train_test(data)
    
            save_as_csv(train_data, 'percentage_' + str(int(percent * 100)) + '_train', 
                        folder=pp + '/percentage/' + str(int(percent * 100)))
            save_as_csv(test_data, 'percentage_' + str(int(percent * 100)) + '_test', 
                        folder=pp + '/percentage/' + str(int(percent * 100)))
        else:
            data = percentage_cutting(data_only_jumps, percent, param)
            save_as_csv(data, 'percentage_' + param + '_' + str(int(percent * 100)),
                        folder=pp + '/percentage/' + str(int(percent * 100)))

            train_data, test_data = split_train_test(data)

            save_as_csv(train_data, 'percentage_' + param + '_' + str(int(percent * 100)) + '_train',
                        folder=pp + '/percentage/' + str(int(percent * 100)))
            save_as_csv(test_data, 'percentage_' + param + '_' + str(int(percent * 100)) + '_test',
                        folder=pp + '/percentage/' + str(int(percent * 100)))
    #"""

    # for vectorisation
    """
    name = 'percentage_mean_std_'           # percentage_mean_std_ / percentage_mean_ / percentage_
    for percent in ['5', '2', '1']:     # '25', '20', '10', '5', '2', '1'
        data = pd.read_csv('Sprungdaten_processed/' + pp + '/percentage/' + percent + '/' + name + percent + '.csv')
        train_data = pd.read_csv(
            'Sprungdaten_processed/' + pp + '/percentage/' + percent + '/' + name + percent + '_train' + '.csv')
        test_data = pd.read_csv('Sprungdaten_processed/' + pp + '/percentage/' + percent + '/' + name + percent + '_test' + '.csv')

        vector_data = vectorize(data)

        vector_train_data = vectorize(train_data)
        vector_test_data = vectorize(test_data)

        save_as_csv(vector_train_data, 'vector_' + name + percent + '_train', folder=pp + '/percentage/' + percent)
        save_as_csv(vector_test_data, 'vector_' + name + percent + '_test', folder=pp + '/percentage/' + percent)
        save_as_csv(vector_data, 'vector_' + name + percent, folder=pp + '/percentage/' + percent)
    #"""
    pp = 'with'
    name = 'mean_'
    for p in ['1']:  # ['25', '20', '10', '5', '2', '1']
        print(p)
        data = pd.read_csv(
            'Sprungdaten_processed/' + pp + '_preprocessed/percentage/' + p + '/vector_percentage_' + name + p + '.csv')
        df = mean_jump_generator(data)
        save_as_csv(df, 'vector_AJ_percentage_' + name + p, folder=pp + '_preprocessed/percentage/' + p)
        devec_df = devectorize(df)
        save_as_csv(devec_df, 'AJ_percentage_' + name + p, folder=pp + '_preprocessed/percentage/' + p)

    # Generating Mean Data and devectoring it
    """
    name = '25'
    pp = 'with'
    data = pd.read_csv('Sprungdaten_processed/' + pp + '_preprocessed/percentage/' + name +'/vector_percentage_' + name + '.csv')
    df = mean_jump_generator(data)
    save_as_csv(df, 'vector_averaged_jumps_percentage_' + name, folder=pp + '_preprocessed/percentage/' + name)
    devec_df = devectorize(df)
    save_as_csv(devec_df, 'averaged_jumps_percentage_' + name, folder=pp + '_preprocessed/percentage/' + name)
    """

    # TODO
    """
    data = pd.read_csv('Sprungdaten_processed/with_preprocessed/data_only_jumps.csv')
    df = create_shap_samples(data)
    save_as_csv(df, 'shap_samples', 'with_preprocessed')
    """



    return


if __name__ == '__main__':
    main()
