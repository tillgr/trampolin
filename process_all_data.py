import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def read_data(name):

    all_data = pd.read_csv("Sprungdaten_processed/" + name + "/" + name + ".csv")

    return all_data


def sort_out_errors(all_data):

    # delete all the unwanted data
    all_data = all_data.query('Sprungtyp != "Datenfehler" & Sprungtyp != "Fehlerhafte Daten" & Sprungtyp != "Unbekannt" & Sprungtyp != "Einturnen"')

    # delete the nan values
    nan_value = float('NaN')
    all_data['Sprungtyp'].replace("", nan_value, inplace=True)
    all_data.dropna(subset=['Sprungtyp'], inplace=True)

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

    data.to_csv('Sprungdaten_processed/' + folder + "/" + name + ".csv", index=False)


def delete_0_point_jumps(data):

    data = data.query('Sprungtyp != "Strecksprung" & Sprungtyp != "Standsprung" & Sprungtyp != "Hocksprung" & Sprungtyp != "Bücksprung" '
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

    return data


def only_marked_jumps(data):
    # TODO

    data = data.query('Sprungtyp == "Fliffis aus C" or Sprungtyp == "Fliffis aus B" or Sprungtyp == "Fliffis- Rudi C" or Sprungtyp == "Fliffis- Rudi B"'
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

    averaged_data = pd.DataFrame(columns=['Sprungtyp', 'SprungID', 'ACC_N', 'ACC_N_ROT_filtered',
                                          'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil', 'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil',
                                          'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                          'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])

    std_data = pd.DataFrame(columns=['Sprungtyp', 'SprungID', 'ACC_N', 'ACC_N_ROT_filtered',
                                     'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil', 'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil'])

    for id in data['SprungID'].unique():

        subframe = data[data['SprungID'] == id]

        mean = subframe.mean()
        std = subframe.std()

        averaged_data = averaged_data.append(mean, ignore_index=True)
        std_data = std_data.append(std, ignore_index=True)

        averaged_data['Sprungtyp'].iloc[len(averaged_data) - 1] = subframe['Sprungtyp'].unique()[0]
        averaged_data['SprungID'].iloc[len(averaged_data) - 1] = subframe['SprungID'].unique()[0]

        std_data['Sprungtyp'].iloc[len(std_data) - 1] = subframe['Sprungtyp'].unique()[0]
        std_data['SprungID'].iloc[len(std_data) - 1] = subframe['SprungID'].unique()[0]

        print(len(averaged_data) - 1)

    averaged_data = averaged_data.drop(columns=['Time', 'TimeInJump'])
    std_data = std_data.drop(columns=['Time', 'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                      'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd', 'TimeInJump'])

    averaged_data = averaged_data.round({'ACC_N': 3, 'ACC_N_ROT_filtered': 3,
                                          'Acc_x_Fil': 3, 'Acc_y_Fil': 3, 'Acc_z_Fil': 3, 'Gyro_x_Fil': 3, 'Gyro_y_Fil': 3, 'Gyro_z_Fil': 3,
                                          'DJump_SIG_I_x LapEnd': 3, 'DJump_SIG_I_y LapEnd': 3, 'DJump_SIG_I_z LapEnd': 3,
                                          'DJump_Abs_I_x LapEnd': 3, 'DJump_Abs_I_y LapEnd': 3, 'DJump_Abs_I_z LapEnd': 3})

    std_data = std_data.round({'ACC_N': 3, 'ACC_N_ROT_filtered': 3, 'Acc_x_Fil': 3, 'Acc_y_Fil': 3, 'Acc_z_Fil': 3,
                               'Gyro_x_Fil': 3, 'Gyro_y_Fil': 3, 'Gyro_z_Fil': 3})

    print(averaged_data)
    print(std_data)
    return averaged_data, std_data


def class_std_mean(avg, std):

    avg_std_data = pd.DataFrame(columns=['Sprungtyp', 'avg_ACC_N', 'avg_ACC_N_ROT_filtered',
                                          'avg_Acc_x_Fil', 'avg_Acc_y_Fil', 'avg_Acc_z_Fil', 'avg_Gyro_x_Fil', 'avg_Gyro_y_Fil', 'avg_Gyro_z_Fil',
                                          'avg_DJump_SIG_I_x LapEnd', 'avg_DJump_SIG_I_y LapEnd', 'avg_DJump_SIG_I_z LapEnd',
                                          'avg_DJump_Abs_I_x LapEnd', 'avg_DJump_Abs_I_y LapEnd', 'avg_DJump_Abs_I_z LapEnd',
                                          'std_ACC_N', 'std_ACC_N_ROT_filtered',
                                          'std_Acc_x_Fil', 'std_Acc_y_Fil', 'std_Acc_z_Fil', 'std_Gyro_x_Fil', 'std_Gyro_y_Fil', 'std_Gyro_z_Fil'])

    for typ in avg['Sprungtyp'].unique():

        avg_subframe = avg[avg['Sprungtyp'] == typ]
        std_subframe = std[std['Sprungtyp'] == typ]

        avg_mean = avg_subframe.mean()
        std_mean = std_subframe.mean()

        avg_mean = avg_mean.to_frame().transpose()
        std_mean = std_mean.to_frame().transpose()
        avg_mean.columns = ['avg_ACC_N', 'avg_ACC_N_ROT_filtered',
                          'avg_Acc_x_Fil', 'avg_Acc_y_Fil', 'avg_Acc_z_Fil', 'avg_Gyro_x_Fil', 'avg_Gyro_y_Fil', 'avg_Gyro_z_Fil',
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

    for (column_name, column_data) in data.iteritems():

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
        df_train, df_test, y_train, y_test = train_test_split(data, data['Sprungtyp'], stratify=data['Sprungtyp'], test_size=0.2, random_state=1)
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
    df_train, df_test, y_train, y_test = train_test_split(df_split, df_split['Sprungtyp'], stratify=df_split['Sprungtyp'], test_size=0.2, random_state=1)
    print("Split of Train/Test Data complete")

    # rewrite dataframes
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    train_data = pd.DataFrame(columns=['Time', 'TimeInJump', 'ACC_N', 'ACC_N_ROT_filtered', 'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil',
                                     'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil', 'SprungID', 'Sprungtyp',
                                     'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                     'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])

    test_data = pd.DataFrame(columns=['Time', 'TimeInJump', 'ACC_N', 'ACC_N_ROT_filtered', 'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil',
                                     'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil', 'SprungID', 'Sprungtyp',
                                     'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                     'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])

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

    avg_data.rename(columns={'ACC_N': 'avg_ACC_N', 'ACC_N_ROT_filtered': 'avg_ACC_N_ROT_filtered',
                             'Acc_x_Fil': 'avg_Acc_x_Fil', 'Acc_y_Fil': 'avg_Acc_y_Fil', 'Acc_z_Fil': 'avg_Acc_z_Fil',
                             'Gyro_x_Fil': 'avg_Gyro_x_Fil', 'Gyro_y_Fil': 'avg_Gyro_y_Fil', 'Gyro_z_Fil': 'avg_Gyro_z_Fil'}, inplace=True)

    std_data.rename(columns={'ACC_N': 'std_ACC_N', 'ACC_N_ROT_filtered': 'std_ACC_N_ROT_filtered',
                             'Acc_x_Fil': 'std_Acc_x_Fil', 'Acc_y_Fil': 'std_Acc_y_Fil', 'Acc_z_Fil': 'std_Acc_z_Fil',
                             'Gyro_x_Fil': 'std_Gyro_x_Fil', 'Gyro_y_Fil': 'std_Gyro_y_Fil', 'Gyro_z_Fil': 'std_Gyro_z_Fil'}, inplace=True)

    avg_std_data = pd.merge(avg_data, std_data, on=['SprungID', 'Sprungtyp'], how='left')
    avg_std_data = avg_std_data[['Sprungtyp', 'SprungID', 'avg_ACC_N', 'std_ACC_N', 'avg_ACC_N_ROT_filtered', 'std_ACC_N_ROT_filtered',
    'avg_Acc_x_Fil', 'std_Acc_x_Fil', 'avg_Acc_y_Fil', 'std_Acc_y_Fil', 'avg_Acc_z_Fil',  'std_Acc_z_Fil',
    'avg_Gyro_x_Fil', 'std_Gyro_x_Fil', 'avg_Gyro_y_Fil', 'std_Gyro_y_Fil', 'avg_Gyro_z_Fil', 'std_Gyro_z_Fil',
    'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
    'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd']]

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
                time_list = np.arange(round(subframe['Time'].iloc[-1] + 0.002, 3), round(subframe['Time'].iloc[-1] + (0.002 * length_to_fill + 0.001), 3), 0.002)
                timeinjump_list = np.arange(round(subframe['TimeInJump'].iloc[-1] + 0.002, 3), round(subframe['TimeInJump'].iloc[-1] + (0.002 * length_to_fill + 0.001), 3), 0.002)
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
                temp_df = pd.DataFrame(data={'SprungID': jump_ids, 'Sprungtyp': jump_typ, 'Time': time_list, 'TimeInJump': timeinjump_list,
                                             'ACC_N': zero_list, 'ACC_N_ROT_filtered': zero_list,
                                             'Acc_x_Fil': zero_list, 'Acc_y_Fil': zero_list, 'Acc_z_Fil': zero_list,
                                             'Gyro_x_Fil': zero_list, 'Gyro_y_Fil': zero_list, 'Gyro_z_Fil': zero_list,
                                             'DJump_SIG_I_x LapEnd': zero_list, 'DJump_SIG_I_y LapEnd': zero_list, 'DJump_SIG_I_z LapEnd': zero_list,
                                             'DJump_Abs_I_x LapEnd': zero_list, 'DJump_Abs_I_y LapEnd': zero_list, 'DJump_Abs_I_z LapEnd': zero_list})

                df = df.append(subframe, ignore_index=True)
                df = df.append(temp_df, ignore_index=True)

    return df


def percentage_cutting(data, percent_steps, method=None):

    if method is None:
        df = pd.DataFrame(columns=['SprungID', 'Sprungtyp', 'Time', 'TimeInJump', 'ACC_N', 'ACC_N_ROT_filtered',
                                   'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil', 'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil',
                                   'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                   'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])
        for jump in data['SprungID'].unique():
            print(jump)
            subframe = data[data['SprungID'] == jump]
            subframe.reset_index(drop=True, inplace=True)
            index_list = np.rint(np.arange(0, len(subframe), len(subframe) * percent_steps))
            if len(index_list) != 100/(percent_steps * 100):
                index_list = index_list[:-1]
            df = df.append(subframe.iloc[index_list], ignore_index=True)
    if method == 'mean':
        df = pd.DataFrame(columns=['SprungID', 'Sprungtyp', 'ACC_N', 'ACC_N_ROT_filtered',
                                       'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil', 'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil',
                                       'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                       'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])
        for jump in data['SprungID'].unique():
            print(jump)
            subframe = data[data['SprungID'] == jump]
            subframe.reset_index(drop=True, inplace=True)
            index_list = np.rint(np.arange(0, len(subframe), len(subframe) * percent_steps))
            if index_list[-1] != len(subframe):
                index_list = np.append(index_list, len(subframe))
            id = subframe['SprungID'][0]
            jump_type = subframe['Sprungtyp'][0]
            for i in range(len(index_list)-1):
                start = int(index_list[i])
                end = int(index_list[i+1])

                """if start == end:
                    temp = subframe.iloc[end-1]
                    temp = temp.to_frame().transpose()
                    temp = temp.drop(['Time', 'TimeInJump'], axis=1)
                else:"""
                temp = subframe.iloc[start:end].mean()
                temp = temp.to_frame().transpose()
                temp = temp.drop(['Time', 'TimeInJump'], axis=1)
                temp.insert(0, 'SprungID', id)
                temp.insert(1, 'Sprungtyp', jump_type)

                df = df.append(temp, ignore_index=True)
    if method == 'mean_std':
        df = pd.DataFrame(columns=['SprungID', 'Sprungtyp',
                                   'mean_ACC_N', 'std_ACC_N',
                                   'mean_ACC_N_ROT_filtered', 'std_ACC_N_ROT_filtered',
                                   'mean_Acc_x_Fil', 'std_Acc_x_Fil',
                                   'mean_Acc_y_Fil', 'std_Acc_y_Fil',
                                   'mean_Acc_z_Fil', 'std_Acc_z_Fil',
                                   'mean_Gyro_x_Fil', 'std_Gyro_x_Fil',
                                   'mean_Gyro_y_Fil', 'std_Gyro_y_Fil',
                                   'mean_Gyro_z_Fil', 'std_Gyro_z_Fil',
                                   'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                   'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])
        for jump in data['SprungID'].unique():
            print(jump)
            subframe = data[data['SprungID'] == jump]
            subframe.reset_index(drop=True, inplace=True)
            index_list = np.rint(np.arange(0, len(subframe), len(subframe) * percent_steps))
            if index_list[-1] != len(subframe):
                index_list = np.append(index_list, len(subframe))
            id = subframe['SprungID'][0]
            jump_type = subframe['Sprungtyp'][0]
            for i in range(len(index_list)-1):
                start = int(index_list[i])
                end = int(index_list[i+1])

                mean = subframe.iloc[start:end].mean()
                std = subframe.iloc[start:end].std()
                mean = mean.to_frame().transpose()
                std = std.to_frame().transpose()
                mean = mean.drop(['Time', 'TimeInJump'], axis=1)
                std = std.drop(['Time', 'TimeInJump', 'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd',
                                'DJump_SIG_I_z LapEnd', 'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd',
                                'DJump_Abs_I_z LapEnd'], axis=1)
                mean.columns = ['mean_ACC_N', 'mean_ACC_N_ROT_filtered',
                                'mean_Acc_x_Fil', 'mean_Acc_y_Fil', 'mean_Acc_z_Fil', 'mean_Gyro_x_Fil',
                                'mean_Gyro_y_Fil', 'mean_Gyro_z_Fil',
                                'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd']
                std.columns = ['std_ACC_N', 'std_ACC_N_ROT_filtered', 'std_Acc_x_Fil', 'std_Acc_y_Fil',
                               'std_Acc_z_Fil', 'std_Gyro_x_Fil', 'std_Gyro_y_Fil', 'std_Gyro_z_Fil']

                temp = pd.concat([mean, std], axis=1)
                temp.insert(0, 'SprungID', id)
                temp.insert(1, 'Sprungtyp', jump_type)

                df = df.append(temp, ignore_index=True)
    return df


def vectorize(data, std=None):
    first = True
    for jump in data['SprungID'].unique():
        print(jump)
        subframe = data[data['SprungID'] == jump]
        subframe.reset_index(drop=True, inplace=True)
        equal_data = subframe[['SprungID', 'Sprungtyp', 'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd',
                               'DJump_SIG_I_z LapEnd', 'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd',
                               'DJump_Abs_I_z LapEnd']].iloc[0]
        equal_data = equal_data.to_frame().transpose()
        percentage_list = np.rint(np.arange(0, 100, 100 / len(subframe)))
        if std is None:
            for row in range(len(subframe)):

                name = str(int(percentage_list[row]))

                copy = subframe[['ACC_N', 'ACC_N_ROT_filtered', 'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil',
                                 'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil']].iloc[int(row)]
                copy = copy.to_frame().transpose()
                copy.columns = [name + '-ACC_N', name + '-ACC_N_ROT_filtered', name + '-Acc_x_Fil',
                                name + '-Acc_y_Fil', name + '-Acc_z_Fil', name + '-Gyro_x_Fil',
                                name + '-Gyro_y_Fil', name + '-Gyro_z_Fil']
                copy.reset_index(drop=True, inplace=True)
                equal_data = pd.concat([equal_data, copy], axis=1)
        else:
            for row in range(len(subframe)):
                name = str(int(percentage_list[row]))
                copy = subframe[['mean_ACC_N', 'std_ACC_N',
                                 'mean_ACC_N_ROT_filtered', 'std_ACC_N_ROT_filtered',
                                 'mean_Acc_x_Fil', 'std_Acc_x_Fil',
                                 'mean_Acc_y_Fil', 'std_Acc_y_Fil',
                                 'mean_Acc_z_Fil', 'std_Acc_z_Fil',
                                 'mean_Gyro_x_Fil', 'std_Gyro_x_Fil',
                                 'mean_Gyro_y_Fil', 'std_Gyro_y_Fil',
                                 'mean_Gyro_z_Fil', 'std_Gyro_z_Fil']].iloc[int(row)]
                copy = copy.to_frame().transpose()
                copy.columns = [name + '-mean_ACC_N', name + '-std_ACC_N',
                                name + '-mean_ACC_N_ROT_filtered', name + '-std_ACC_N_ROT_filtered',
                                name + '-mean_Acc_x_Fil', name + '-std_Acc_x_Fil',
                                name + '-mean_Acc_y_Fil', name + '-std_Acc_y_Fil',
                                name + '-mean_Acc_z_Fil', name + '-std_Acc_z_Fil',
                                name + '-mean_Gyro_x_Fil', name + '-std_Gyro_x_Fil',
                                name + '-mean_Gyro_y_Fil', name + '-std_Gyro_y_Fil',
                                name + '-mean_Gyro_z_Fil', name + '-std_Gyro_z_Fil']
                copy.reset_index(drop=True, inplace=True)
                equal_data = pd.concat([equal_data, copy], axis=1)
        if first is True:
            df = pd.DataFrame(columns=equal_data.columns)
            first = False
        df = df.append(equal_data, ignore_index=True)
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

    """
    averaged_data = read_data("averaged_data")
    std_data = read_data("std_data")
    avg_std_data = combine_avg_std(averaged_data, std_data)
    save_as_csv(avg_std_data, "avg_std_data", folder="avg_std_data")
    """

    """
    for method in ['cut_first', 'cut_last', 'padding_0']:
        data_point_jumps = read_data("data_point_jumps")
        data = make_jump_length_consistent(data_point_jumps, method=method)
        save_as_csv(data, "same_length_" + method, folder="same_length")
    """
    # template for creating datasets percentage
    """
    data_point_jumps = read_data("data_point_jumps")
    # percentage_cutting with 'mean' , 'mean_std' or nothing
    data = percentage_cutting(data_point_jumps, 0.02, 'mean_std')
    save_as_csv(data, 'percentage_mean_std_2', folder='percentage/2')
    train_data, test_data = split_train_test(data)
    save_as_csv(train_data, 'percentage_mean_std_2_train', folder='percentage/2')
    save_as_csv(test_data, 'percentage_mean_std_2_test', folder='percentage/2')
    """

    # template for vectorisation
    '''
    name = 'percentage_1'
    data_point_jumps = pd.read_csv('Sprungdaten_processed/percentage/1/' + name + '.csv')
    data = vectorize(data_point_jumps)
    train_data, test_data = split_train_test(data)
    save_as_csv(train_data, 'vector_' + name + '_train', folder='percentage/1')
    save_as_csv(test_data,  'vector_' + name + '_test', folder='percentage/1')
    save_as_csv(data, 'vector_' + name, folder='percentage/1')
    '''

    #"""
    name = 'percentage_mean_std_'
    for percent in ['25', '20', '10', '5', '2', '1']:
        data = pd.read_csv('Sprungdaten_processed/percentage/' + percent + '/' + name + percent + '.csv')
        train_data = pd.read_csv('Sprungdaten_processed/percentage/' + percent + '/' + name + percent + '_train' + '.csv')
        test_data = pd.read_csv('Sprungdaten_processed/percentage/' + percent + '/' + name + percent +'_test'+ '.csv')
        if 'mean' not in name:
            data = data.drop(columns=['Time', 'TimeInJump'])
            train_data = train_data.drop(columns=['Time', 'TimeInJump'])
            test_data = test_data.drop(columns=['Time', 'TimeInJump'])

        if 'std' in name:
            vector_data = vectorize(data, 'std')

            vector_train_data = vectorize(train_data, 'std')
            vector_test_data = vectorize(test_data, 'std')
        else:
            vector_data = vectorize(data)

            vector_train_data = vectorize(train_data)
            vector_test_data = vectorize(test_data)
        save_as_csv(vector_train_data, 'vector_' + name + percent + '_train', folder='percentage/' + percent)
        save_as_csv(vector_test_data,  'vector_' + name + percent + '_test', folder='percentage/' + percent)
        save_as_csv(vector_data, 'vector_' + name + percent, folder='percentage/' + percent)
    #"""

    """
    data_point_jumps = read_data("data_point_jumps")
    param = 'mean_std'
    for percent in [0.25, 0.20, 0.10, 0.05, 0.02, 0.01]:
        data = percentage_cutting(data_point_jumps, percent, param)
        save_as_csv(data, 'percentage_' + param + '_' + str(int(percent * 100)), folder='percentage/' + str(int(percent * 100)))
        train_data, test_data = split_train_test(data)
        if param in ['mean', 'mean_std']:
            train_data = train_data.drop(columns=['Time', 'TimeInJump'])
            test_data = test_data.drop(columns=['Time', 'TimeInJump'])
        save_as_csv(train_data, 'percentage_' + param + '_' + str(int(percent * 100)) + '_train', folder='percentage/' + str(int(percent * 100)))
        save_as_csv(test_data, 'percentage_' + param + '_' + str(int(percent * 100)) + '_test', folder='percentage/' + str(int(percent * 100)))

    """

    return


if __name__ == '__main__':
    main()
