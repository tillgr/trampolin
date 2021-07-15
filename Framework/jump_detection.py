import numpy as np
import pandas as pd
import Framework.process_data as process
import datetime


def cost(threshold: int, y):
    """
    Calculates the number of jumps that would currently be found with the threshold

    :param threshold: int - Current threshold to test
    :param y: array - ACC_N_Fil Column of data
    :return: number of jumps found
    """

    count = 0
    for i in range(len(y) - 100):
        i += 50
        if y[i] > threshold:
            if all(l < y[i] for l in y[i - 50:i]):
                if all(l < y[i] for l in y[i + 1:i + 50]):
                    count += 1
    print(count)
    return count


def estimate_threshhold(threshold: int, jumps_amount, y):
    """
    Calculates a threshold, which can find the correct amount of jumps for a given dataset

    threshold: int - Current threshold to test
    :param jumps_amount: int - Number of jumps to find
    :param y: array - ACC_N_Fil Column of data
    :return: correct threshold
    """

    threshold_cost = cost(threshold, y)
    while threshold_cost != jumps_amount:
        if threshold_cost > jumps_amount:
            threshold += ((abs(threshold_cost - jumps_amount)) / 100)
            threshold_cost = cost(threshold, y)
        else:
            threshold -= ((abs(threshold_cost - jumps_amount)) / 100)
            threshold_cost = cost(threshold, y)
    return threshold


def find_threshold(jumps_to_classify):
    """
    Finds a threshold bottom-up and top-down

    :param jumps_to_classify: int - number of jumps in data. For all jumps use len(data['SprungID'].unique())
    """

    thresholdMin = 40
    thresholdMax = 100
    data = pd.read_csv("Sprungdaten_processed/without_preprocessed/data_only_jumps.csv")
    last_index = np.where(data['SprungID'] == data['SprungID'].unique()[jumps_to_classify - 1])[0][-1]
    data = data[:last_index]
    jumps_amount = len(data['SprungID'].unique())
    x = np.array((data['Time']))
    y = np.array((data['Acc_N_Fil']))
    z = np.array((data['SprungID']))

    y = np.pad(y, (50, 50), 'constant')

    thresholdMin = estimate_threshhold(thresholdMin, jumps_amount, y)  # 64.34999999999997
    thresholdMax = estimate_threshhold(thresholdMax, jumps_amount, y)  # 64.42999999999999

    print("With ThresholdMin:")
    for i in range(len(y) - 100):
        i += 50
        if y[i] > thresholdMax:
            if all(l < y[i] for l in y[i - 50:i]):
                if all(l < y[i] for l in y[i + 1:i + 50]):
                    print(f" index: {i - 50}:, time: {x[i - 100]}, acceleration: {y[i]}, id: {z[i - 100]}")

    print("")
    print("With ThresholdMin:")

    for i in range(len(y) - 100):
        i += 50
        if y[i] > thresholdMin:
            if all(l < y[i] for l in y[i - 50:i]):
                if all(l < y[i] for l in y[i + 1:i + 50]):
                    print(f" index: {i - 50}:, time: {x[i - 100]}, acceleration: {y[i]}, id: {z[i - 100]}")
    print(f"the threshold is between {thresholdMin} and {thresholdMax}")


def detect_jump_starts(data):
    """
    Detects the start of each jump so we can then classify all the jumps in the data

    :param data: dataframe - read from csv file
    :return: dataframe with id, segmented by jumps
    """

    threshold = 64.4

    data = data.reset_index(drop=True)

    if 'Dist' in data.columns:
        data = data.drop(['Dist'], axis=1)

    for col in data.columns:
        try:
            data[col] = data[col].apply(process.convert_comma_to_dot)
        except:
            pass

    a = np.array((data['Acc_N_Fil']))
    a = np.pad(a, (50, 50), 'constant')

    index_list = []

    for i in range(len(a) - 100):
        i += 50
        if a[i] > threshold:
            if all(l < a[i] for l in a[i - 50:i]):
                if all(l < a[i] for l in a[i + 1:i + 50]):
                    print(f" index: {i - 50}: acceleration: {a[i]}")
                    index_list.append(i - 50)

    print(index_list)

    for i in range(len(index_list) - 1):
        time = str(datetime.datetime.now())
        for x in range(index_list[i], index_list[i + 1]):
            data.loc[data.index[x], 'SprungID'] = time + "-" + str(i)

    data = data.dropna()

    return data
