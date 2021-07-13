import numpy as np
import pandas as pd


def number_of_jumps(data):
    jump_list = []
    for i in range(10000):
        jump_list.append(data[i])
    jumps = list(dict.fromkeys(jump_list))
    return print(len(jumps))


def data_size(data):
    jump_list = []
    for i in range(10000):
        jump_list.append(data[i])
    print(len(jump_list))


def cost(threshold: int):
    count = 0
    for i in range(10000):
        if y[i] > y[i + 1] and y[i] > y[i - 1] and y[i] > threshold:
            count += 1
            if count > 10:
                break
    return count


def estimate_threshhold(threshold: int):
    while cost(threshold) != jumps_amount:
        if cost(threshold) > jumps_amount:
            threshold += 1
            cost(threshold)
        else:
            threshold -= 1
            cost(threshold)
    return threshold


if __name__ == '__main__':
    # jumps_amount = 5927 / 5463628 # jumps amount = 101 / 100000
    # jumps_amount = 3 / 2715 # jumps_amount = 1072 / 1000000 # all = 5463628 # jumps_amount = 10 / 10000
    thresholdMin = 0
    thresholdMax = 100
    jumps_amount = 10
    data = pd.read_csv("../sgd/all_data_new.csv")
    x = np.array((data['Time']))
    y = np.array((data['Acc_N_Fil']))
    z = np.array((data['SprungID']))
    # plt.plot(x[0:2700], y[0:2700])
    # plt.show()
    # number_of_jumps(z);
    thresholdMin = estimate_threshhold(thresholdMin)
    thresholdMax = estimate_threshhold(thresholdMax)

    for i in range(10000):
        if y[i] > y[i + 1] and y[i] > y[i - 1] and y[i] > thresholdMax:
            print(f" index: {i + 2}:, time: {x[i]}, acceleration: {y[i]}, id: {z[i]}")
    print(f"the threshold is between {thresholdMin} and {thresholdMax}")
