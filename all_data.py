import numpy as np
import pandas as pd
import os


def read_csv_data(file):

    raw_data = pd.read_csv(file, header=None, skiprows=2, delimiter=";")
    raw_data.columns = ["Time", "Dist", "TimeInJump", "ACC_N", "ACC_N_ROT_filtered", "Acc_x_Fil", "Acc_y_Fil", "Acc_z_Fil", "Gyro_x_Fil", "Gyro_y_Fil", "Gyro_z_Fil"]
    print(file)

    return raw_data


def read_xlsx_data(file):

    raw_data = pd.read_excel(file, engine="openpyxl")
    print(file)

    return raw_data


def convert_to_numeric(x):
    if "sec" in x:

        return float(x.replace(' sec', ''))

    if "min" in x:
        x = x.replace(" min", "")
        times = x.split(":")
        minutes = float(times[0])
        seconds = round(float(times[1]), 3)

        return float(minutes * 60.0 + seconds)


def convert_comma_to_dot(x):
    return float(x.replace(',', '.'))


def main():

    all_data = pd.DataFrame(columns=['Time', 'TimeInJump', 'ACC_N', 'ACC_N_ROT_filtered', 'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil',
                                     'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil', 'SprungID', 'Sprungtyp',
                                     'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                     'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])

    folders = os.listdir('Sprungdaten Innotramp')
    for folder in folders:
        files = os.listdir("Sprungdaten Innotramp/" + folder)
        for i in range(len(files[:int(len(files) / 2)])):           # we can read the corresponding sprungzuordnung and rohdaten this way
            temp_data = pd.DataFrame(columns=['Time', 'TimeInJump', 'ACC_N', 'ACC_N_ROT_filtered', 'Acc_x_Fil', 'Acc_y_Fil', 'Acc_z_Fil',
                         'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil'])

            csv_data = read_csv_data("Sprungdaten Innotramp/" + folder + "/" + files[i])
            xlsx_data = read_xlsx_data("Sprungdaten Innotramp/" + folder + "/" + files[i + int(len(files) / 2)])

            csv_data['Time'] = np.round(csv_data['Time'].apply(convert_comma_to_dot), 3)
            xlsx_data["Zeit"] = xlsx_data["Zeit"].apply(convert_to_numeric)

            temp_data = temp_data.append(csv_data.drop(['Dist'], axis=1))

            cumultime = np.cumsum(xlsx_data['Zeit'].to_numpy(), dtype=float)        # cumultative sum to make the time equal to the rohdaten
            xlsx_data['cumultime'] = cumultime

            start_time = csv_data['Time'][0]

            sprungzuordnung = pd.DataFrame(columns=['Time', 'SprungID', 'Sprungtyp',
                         'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                         'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])

            for row in xlsx_data.iterrows():

                end_time = row[1]['cumultime']
                join_times = np.arange(start_time, end_time, 0.002)                 # creates all times with 0.002 steps from start to end
                SprungID = row[1]['Messung'] + "-" + str(row[1]['Lap#'])            # create an unique ID for each jump

                multiply_array = np.array([SprungID, row[1]['Sprungtyp'], row[1]['DJump_SIG_I_x LapEnd'], row[1]['DJump_SIG_I_y LapEnd'], row[1]['DJump_SIG_I_z LapEnd'],
                                           row[1]['DJump_Abs_I_x LapEnd'], row[1]['DJump_Abs_I_y LapEnd'], row[1]['DJump_Abs_I_z LapEnd']])
                multiply_array = np.transpose(np.repeat(multiply_array.reshape(8, 1), len(join_times), axis=1))         # this array will be multiplied when joined with the rohdaten

                temp_sprungzuordnung = pd.DataFrame(multiply_array, columns=['SprungID', 'Sprungtyp', 'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                                      'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'])
                temp_sprungzuordnung['Time'] = np.round(join_times, 3)

                sprungzuordnung = sprungzuordnung.append(temp_sprungzuordnung)

                start_time = end_time

            temp_data = pd.merge(temp_data, sprungzuordnung, on='Time', how='left')

            all_data = all_data.append(temp_data, ignore_index=True)

    all_data['TimeInJump'] = all_data['TimeInJump'].apply(convert_comma_to_dot)
    all_data['ACC_N'] = all_data['ACC_N'].apply(convert_comma_to_dot)
    all_data['ACC_N_ROT_filtered'] = all_data['ACC_N_ROT_filtered'].apply(convert_comma_to_dot)
    all_data['Acc_x_Fil'] = all_data['Acc_x_Fil'].apply(convert_comma_to_dot)
    all_data['Acc_y_Fil'] = all_data['Acc_y_Fil'].apply(convert_comma_to_dot)
    all_data['Acc_z_Fil'] = all_data['Acc_z_Fil'].apply(convert_comma_to_dot)
    all_data['Gyro_x_Fil'] = all_data['Gyro_x_Fil'].apply(convert_comma_to_dot)
    all_data['Gyro_y_Fil'] = all_data['Gyro_y_Fil'].apply(convert_comma_to_dot)
    all_data['Gyro_z_Fil'] = all_data['Gyro_z_Fil'].apply(convert_comma_to_dot)

    all_data.to_csv('Sprungdaten_processed/all_data.csv', index=False)

    return


if __name__ == '__main__':
    main()
