import numpy as np
import pandas as pd
import os


def read_raw_data(file):

    raw_data = pd.read_csv(file, header=None, skiprows=2, delimiter=";")
    raw_data.columns = ["Time", "Dist", "TimeInJump", "ACC_N", "ACC_N_ROT_filtered", "Acc_x_Fil", "Acc_y_Fil", "Acc_z_Fil", "Gyro_x_Fil", "Gyro_y_Fil", "Gyro_z_Fil"]

    return raw_data


def find_missing_values(raw_data):

    timestamps = raw_data["Time"].to_numpy()
    time_in_jump = raw_data["TimeInJump"].to_numpy()

    missing = []

    for i in range(len(timestamps) - 1):
        current = float(timestamps[i].replace(',', '.'))
        next = float(timestamps[i + 1].replace(',', '.'))
        if next == round(current + 0.002, 3):
            pass
        else:
            if time_in_jump[i + 1] != "0,002":
                missing.append(round(current + 0.002, 3))

    print(len(missing))
    return missing


def main():

    folders = os.listdir('Sprungdaten Innotramp')
    for folder in folders:
        files = os.listdir("Sprungdaten Innotramp/" + folder)
        for file in files:
            if file.endswith(".csv"):
                raw_data = read_raw_data("Sprungdaten Innotramp/" + folder + "/" + file)
                find_missing_values(raw_data)

    return


if __name__ == '__main__':
    main()
