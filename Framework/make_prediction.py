import Framework.neural_networks as nn
import Framework.process_data as process
import Framework.jump_detection as jd
import keras
import pandas as pd
import os


if __name__ == '__main__':

    best_model_with = "CNN_with_mean_std_20"
    best_model_without = "DFF_without_mean_std_20"
    best_model_with_percentage = 0.2
    best_model_without_percentage = 0.2
    best_model_with_method = 'mean_std'
    best_model_without_method = 'mean_std'
    best_model_with_pp_list = [3]

    # with or without preprocessed
    print("Please make sure that the data you want to make a prediction on are the only files located in Prediction data!")
    print("Does the data contain preprocessed data? [y/n]")
    x = input()

    if x == 'y' or x == 'Y':
        model = keras.models.load_model("models/" + best_model_with)

    elif x == 'n' or x == 'N':
        model = keras.models.load_model("models/" + best_model_without)

    else:
        raise AttributeError("You did not specify whether the data contains preprocessed data or not")

    files = os.listdir("Prediction data/")
    for file in files:
        data = pd.read_csv("Prediction data/" + file)

        data = jd.detect_jump_starts(data)

        data['Sprungtyp'] = ""

        if x == 'y' or x == 'Y':
            df = process.percentage_cutting(data, best_model_with_percentage, best_model_with_method)
            df = df.drop(['Sprungtyp'], axis=1)
            pred = nn.predict_CNN(model, df, best_model_with_pp_list)
        elif x == 'n' or x == 'N':
            df = process.percentage_cutting(data, best_model_without_percentage, best_model_without_method)
            data = data.drop(['Sprungtyp'], axis=1)
            pred = nn.predict_DFF(model, df, [])

        print("Prediction for file " + file + ": " + str(pred))
