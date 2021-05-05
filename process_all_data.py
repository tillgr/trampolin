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

    for column in list(data.columns.values):
        try:
            data = data.round({column: 3})
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

    for file in ["jumps_time_splits"]:
        data_point_jumps = read_data(file)
        train_data, test_data = split_train_test(data_point_jumps)
        save_as_csv(train_data, file + "_train", folder=file)
        save_as_csv(test_data, file + "_test", folder=file)

    return


if __name__ == '__main__':
    main()
