import numpy as np
import pandas as pd
import os


def read_csv_data(file):

    raw_data = pd.read_csv(file, header=None, skiprows=2, delimiter=";")
    raw_data.columns = pd.read_csv(file, nrows=0, delimiter=";").columns
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
    first_round = True
    first_temp = True
    global_xlsx_col_names = []

    folders = os.listdir('Sprungdaten Innotramp')
    folders = ['2019.09.30']
    for folder in folders:
        files = os.listdir("Sprungdaten Innotramp/" + folder)
        for i in range(len(files[:int(len(files) / 2)])):           # we can read the corresponding sprungzuordnung and rohdaten this way

            csv_data = read_csv_data("Sprungdaten Innotramp/" + folder + "/" + files[i])
            xlsx_data = read_xlsx_data("Sprungdaten Innotramp/" + folder + "/" + files[i + int(len(files) / 2)])

            csv_data['Time'] = np.round(csv_data['Time'].apply(convert_comma_to_dot), 3)    # convert time stucture
            xlsx_data["Zeit"] = xlsx_data["Zeit"].apply(convert_to_numeric)

            temp_data = pd.DataFrame(columns=csv_data.drop(['Dist'], axis=1).columns)   # df for csv data

            temp_data = temp_data.append(csv_data.drop(['Dist'], axis=1))

            cumultime = np.cumsum(xlsx_data['Zeit'].to_numpy(), dtype=float)        # cumultative sum to make the time equal to the rohdaten
            xlsx_data['cumultime'] = cumultime

            start_time = csv_data['Time'][0]

            for row in xlsx_data.iterrows():

                end_time = row[1]['cumultime']
                join_times = np.arange(start_time, end_time, 0.002)                 # creates all times with 0.002 steps from start to end
                sprungID = row[1]['Messung'] + "-" + str(row[1]['Lap#'])            # create an unique ID for each jump


                multiply_array = np.array([sprungID, row[1]['Sprungtyp']])
                for col in xlsx_data.columns:
                    if 'DJump' in col:
                        multiply_array = np.append(multiply_array, row[1][col])
                multiply_array = np.transpose(np.repeat(multiply_array.reshape(len(multiply_array), 1), len(join_times), axis=1))         # this array will be multiplied when joined with the rohdaten
                if first_temp:                                                      # for fixing lower case problem in xlsx column names
                    global_xlsx_col_names = ['SprungID', 'Sprungtyp'] + [col for col in xlsx_data.columns if 'DJump' in col]
                temp_sprungzuordnung = pd.DataFrame(multiply_array, columns=global_xlsx_col_names)
                temp_sprungzuordnung['Time'] = np.round(join_times, 3)
                if first_temp:
                    sprungzuordnung = pd.DataFrame(columns=temp_sprungzuordnung.columns)

                    first_temp = False
                sprungzuordnung = sprungzuordnung.append(temp_sprungzuordnung)

                start_time = end_time

            temp_data = pd.merge(temp_data, sprungzuordnung, on='Time', how='left')
            temp_data.drop(temp_data[temp_data['Sprungtyp'] == 'nan'].index, inplace=True)
            if first_round:
                all_data = pd.DataFrame(columns=temp_data.columns)
                first_round = False
            all_data = all_data.append(temp_data, ignore_index=True)
            sprungzuordnung = sprungzuordnung[0:0]

        for col in all_data.columns:
            if col in ['Acc_N_Fil', 'Gyro_x_R', 'Gyro_y_R', 'Gyro_z_R', 'Gyro_x_Fil', 'Gyro_y_Fil', 'Gyro_z_Fil']:
                all_data[col] = all_data[col].apply(convert_comma_to_dot)

        all_data.to_csv('Sprungdaten_processed/all_data_' + folder + '.csv', index=False)
        all_data = all_data[0:0]

    return



if __name__ == '__main__':
    main()
