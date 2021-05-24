import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("../Sprungdaten_processed/all_data.csv")

    x = np.array((data['Time']))
    y = np.array((data['Acc_x_Fil']))
    plt.plot(x[0:10000], y[0:10000])
    plt.show()

    for i in range(len(y[0:10000])):
        if y[i - 1] < 5 and y[i] >= 5 and y[i + 1] > 5:
            print(f" index: {i + 2}:, {y[i]}, {y[i - 1]}, {y[i + 1]} ")
            for n in reversed(range(len(y[0:i]))):
                if y[n] < y[n+1] and y[n] < y[n-1]:
                    print(f" Time: {x[n]}, Jump beginning: {y[n]}")
                    break
