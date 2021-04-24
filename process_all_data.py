import numpy as np
import pandas as pd


def read_all_data():

    all_data = pd.read_csv("Sprungdaten_processed/all_data.csv")

    return all_data


def sort_out_errors(all_data):

    all_data = all_data.query('Sprungtyp != "Datenfehler" & Sprungtyp != "Fehlerhafte Daten" & Sprungtyp != "Unbekannt" & Sprungtyp != "Einturnen"')

    nan_value = float('NaN')
    all_data['Sprungtyp'].replace("", nan_value, inplace=True)
    all_data.dropna(subset=['Sprungtyp'], inplace=True)

    all_data = all_data.reset_index(drop=True)

    return all_data


def save_as_csv(data, name):

    data['Time'] = np.round(data['Time'], 3)

    data.to_csv('Sprungdaten_processed/' + name, index=False)


def main():

    all_data = read_all_data()
    data_only_jumps = sort_out_errors(all_data)

    save_as_csv(data_only_jumps, 'data_only_jumps.csv')

    return


if __name__ == '__main__':
    main()
