import numpy as np
import pandas as pd


def read_data(name):

    all_data = pd.read_csv("Sprungdaten_processed/" + name)

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


def save_as_csv(data, name, with_time=True):

    if with_time:
        data['Time'] = np.round(data['Time'], 3)

    data.to_csv('Sprungdaten_processed/' + name, index=False)


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

    data['ACC_N'] = data['ACC_N'].apply(convert_comma_to_dot)
    data['ACC_N_ROT_filtered'] = data['ACC_N_ROT_filtered'].apply(convert_comma_to_dot)
    data['Acc_x_Fil'] = data['Acc_x_Fil'].apply(convert_comma_to_dot)
    data['Acc_y_Fil'] = data['Acc_y_Fil'].apply(convert_comma_to_dot)
    data['Acc_z_Fil'] = data['Acc_z_Fil'].apply(convert_comma_to_dot)
    data['Gyro_x_Fil'] = data['Gyro_x_Fil'].apply(convert_comma_to_dot)
    data['Gyro_y_Fil'] = data['Gyro_y_Fil'].apply(convert_comma_to_dot)
    data['Gyro_z_Fil'] = data['Gyro_z_Fil'].apply(convert_comma_to_dot)

    for id in data['SprungID'].unique():

        subframe = data[data['SprungID'] == id]

        mean = subframe.mean()

        averaged_data = averaged_data.append(mean, ignore_index=True)

        averaged_data['Sprungtyp'].iloc[len(averaged_data) - 1] = subframe['Sprungtyp'].unique()[0]
        averaged_data['SprungID'].iloc[len(averaged_data) - 1] = subframe['SprungID'].unique()[0]

        print(len(averaged_data) - 1)

    averaged_data = averaged_data.drop(columns=['Time'])

    print(averaged_data)
    return averaged_data


def main():

    """
    all_data = read_all_data("all_data.csv")
    data_only_jumps = sort_out_errors(all_data)

    count_jumptypes = data_only_jumps['Sprungtyp'].value_counts()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(count_jumptypes)

    # save_as_csv(data_only_jumps, 'data_only_jumps.csv')
    """

    """
    data_only_jumps = read_data("data_only_jumps.csv")
    data_point_jumps = delete_0_point_jumps(data_only_jumps)
    save_as_csv(data_point_jumps, "data_point_jumps.csv")
    """

    """
    data_point_jumps = read_data("data_point_jumps.csv")
    data_point_jumps = correct_space_errors(data_point_jumps)
    print(data_point_jumps['Sprungtyp'].unique())
    save_as_csv(data_point_jumps, "data_point_jumps.csv")
    """

    """
    data_point_jumps = read_data("data_point_jumps.csv")
    marked_jumps = only_marked_jumps(data_point_jumps)
    """

    data_point_jumps = read_data("data_point_jumps.csv")
    averaged_data = calc_avg(data_point_jumps)
    save_as_csv(averaged_data, "averaged_data.csv", with_time=False)

    return


if __name__ == '__main__':
    main()
