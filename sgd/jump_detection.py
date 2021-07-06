import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cost(threshold: int):
    count = 0
    for i in range(2700):
        if y[i] > y[i + 1] and y[i] > y[i - 1] and y[i] > threshold:
            count += 1
            if count > 10:
                break
    return count


if __name__ == '__main__':
    numberOfJumps = 2
    threshold = 0
    data = pd.read_csv("../sgd/all_data_new.csv")
    x = np.array((data['Time']))
    y = np.array((data['Acc_N_Fil']))
    z = np.array((data['Acc_N_Fil']))
    plt.plot(x[0:2700], y[0:2700])
    plt.show()

    '''for i in range(len(y[0:10000])):
        if y[i - 1] < 5 and y[i] >= 5 and y[i + 1] > 5:
            print(f" index: {i + 2}:, {y[i]}, {y[i - 1]}, {y[i + 1]} ")
            for n in reversed(range(len(y[0:i]))):
                if y[n] < y[n+1] and y[n] < y[n-1]:
                    print(f" Time: {x[n]}, Jump beginning: {y[n]}")
                    break'''

    while cost(threshold) != numberOfJumps:
        if cost(threshold) > numberOfJumps:
            threshold += 1
            cost(threshold)
        else:
            threshold -= 1
            cost(threshold)

    for i in range(2700):
        if y[i] > y[i + 1] and y[i] > y[i - 1] and y[i] > threshold:
            print(f" index: {i + 2}:, time: {x[i]}, {y[i]}")
    print(threshold)
