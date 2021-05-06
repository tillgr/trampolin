import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten
from keras.utils import to_categorical
from keras import backend as k


def prepare_data():

    # data_train = pd.read_csv("Sprungdaten_processed/same_length/same_length_cut_last_train.csv")
    # data_test = pd.read_csv("Sprungdaten_processed/same_length/same_length_cut_last_test.csv")
    data_train = pd.read_csv("Sprungdaten_processed/percentage/5/percentage_5_train.csv")
    data_test = pd.read_csv("Sprungdaten_processed/percentage/5/percentage_5_test.csv")

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    jump_data_length = len(data_train[data_train['SprungID'] == data_train['SprungID'].unique()[0]])

    for id in data_train['SprungID'].unique():
        subframe = data_train[data_train['SprungID'] == id]
        y_train.append(subframe['Sprungtyp'].unique()[0])
        subframe = subframe.drop(['Time', 'TimeInJump', 'SprungID', 'Sprungtyp'], axis=1)
        x_train.append(subframe)
        print("Preparing train data: " + str(len(x_train)))

    for id in data_test['SprungID'].unique():
        subframe = data_test[data_test['SprungID'] == id]
        y_test.append(subframe['Sprungtyp'].unique()[0])
        subframe = subframe.drop(['Time', 'TimeInJump', 'SprungID', 'Sprungtyp'], axis=1)
        x_test.append(subframe)
        print("Preparing test data: " + str(len(x_test)))

    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], jump_data_length, 14, 1)
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], jump_data_length, 14, 1)

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    return x_train, y_train, x_test, y_test, jump_data_length


def build_model(p, q, jump_data_length):

    first_input = Input(shape=(jump_data_length, 14, 1), name="first_input")
    x = Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu")(first_input)
    for i in range(p):
        x = Conv2D(32 * (i + 1), kernel_size=(1 + q, 1 + q), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(q, q), padding="same")(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(41, activation='softmax', name="output")(x)

    model = Model(inputs=first_input, outputs=x)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model


def grid_seach_build(x_train, y_train, x_test, y_test, jump_data_length):

    best_score = 0
    best_model = 0

    for p in range(4):
        for q in range(1, 4):
            print("Training model with p = " + str(p) + " and q = " + str(q))
            model = build_model(p, q, jump_data_length)
            model.fit(x_train, y_train, batch_size=32, epochs=12, verbose=1)
            score = model.evaluate(x_test, y_test, verbose=1)
            if score[1] > best_score:
                best_score = score[1]
                best_model = model
                model_parameters = [p, q]
                print("Best model with: p = " + str(model_parameters[0]) + " and q = " + str(model_parameters[1]) + " with accuracy = " + str(best_score))
    print("Best model with: p = " + str(model_parameters[0]) + " and q = " + str(model_parameters[1]) + " with accuracy = " + str(best_score))

    return best_model


def main():

    x_train, y_train, x_test, y_test, jump_data_length = prepare_data()
    model = grid_seach_build(x_train, y_train, x_test, y_test, jump_data_length)
    model.evaluate(x_test, y_test, verbose=1)

    return


if __name__ == '__main__':
    main()
