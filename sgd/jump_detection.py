import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cost(threshold: int):
    count = 0
    for i in range(len(y) - 100):
        i += 50
        if y[i] > threshold:
            if all(l < y[i] for l in y[i - 50:i]):
                if all(l < y[i] for l in y[i + 1:i + 50]):
                    count += 1
    print(count)
    return count


def estimate_threshhold(threshold: int):
    threshold_cost = cost(threshold)
    while threshold_cost != jumps_amount:
        if threshold_cost > jumps_amount:
            threshold += ((abs(threshold_cost - jumps_amount)) / 100)
            threshold_cost = cost(threshold)
        else:
            threshold -= ((abs(threshold_cost - jumps_amount)) / 100)
            threshold_cost = cost(threshold)
    return threshold


if __name__ == '__main__':
    # jumps_amount = 5927 / 5463628 # jumps amount = 101 / 100000
    # jumps_amount = 3 / 2715 # jumps_amount = 1072 / 1000000 # all = 5463628 # jumps_amount = 10 / 10000
    thresholdMin = 40
    thresholdMax = 100
    data = pd.read_csv("../Sprungdaten_processed/without_preprocessed/data_only_jumps.csv")
    jumps_to_classify = 100     # len(data['SprungID'].unique())
    last_index = np.where(data['SprungID'] == data['SprungID'].unique()[jumps_to_classify - 1])[0][-1]
    data = data[:last_index]
    jumps_amount = len(data['SprungID'].unique())
    x = np.array((data['Time']))
    y = np.array((data['Acc_N_Fil']))
    z = np.array((data['SprungID']))

    y = np.pad(y, (50, 50), 'constant')

    #plt.plot(y[50:len(y) - 50])
    #plt.show()
    thresholdMin = estimate_threshhold(thresholdMin)        # 64
    thresholdMax = estimate_threshhold(thresholdMax)        # 64

    print("With ThreshholdMin:")
    for i in range(len(y) - 100):
        i += 50
        if y[i] > thresholdMax:
            if all(l < y[i] for l in y[i - 50:i]):
                if all(l < y[i] for l in y[i + 1:i + 50]):
                    print(f" index: {i - 50}:, time: {x[i - 100]}, acceleration: {y[i]}, id: {z[i - 100]}")

    print("")
    print("With ThreshholdMin:")

    for i in range(len(y) - 100):
        i += 50
        if y[i] > thresholdMin:
            if all(l < y[i] for l in y[i - 50:i]):
                if all(l < y[i] for l in y[i + 1:i + 50]):
                    print(f" index: {i - 50}:, time: {x[i - 100]}, acceleration: {y[i]}, id: {z[i - 100]}")
    print(f"the threshold is between {thresholdMin} and {thresholdMax}")
