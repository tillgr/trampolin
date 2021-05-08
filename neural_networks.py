import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Dropout, SpatialDropout2D, AveragePooling2D
from keras import backend as k


def prepare_data():

    # data_train = pd.read_csv("Sprungdaten_processed/same_length/same_length_padding_0_train.csv")
    # data_test = pd.read_csv("Sprungdaten_processed/same_length/same_length_padding_0_test.csv")
    data_train = pd.read_csv("Sprungdaten_processed/percentage/5/percentage_mean_5_train.csv")
    data_test = pd.read_csv("Sprungdaten_processed/percentage/5/percentage_mean_5_test.csv")

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    num_columns = 14

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
    x_train = x_train.reshape(x_train.shape[0], jump_data_length, num_columns, 1)
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], jump_data_length, num_columns, 1)

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    return x_train, y_train, x_test, y_test, jump_data_length, num_columns


def build_model_testing(jump_data_length, num_columns):

    first_input = Input(shape=(jump_data_length, num_columns, 1), name="first_input")
    x = Conv2D(64, kernel_size=(3, 3), padding="same", activation='tanh')(first_input)
    x = Conv2D(64, kernel_size=(3, 3), padding="same", activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), padding="same", activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), padding="same", activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='tanh')(x)
    x = Dense(41, activation='softmax', name="output")(x)

    model = Model(inputs=first_input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

    return model


def build_model(jump_data_length, num_columns, c, kernel, pool, d, act_func, loss, optim):

    first_input = Input(shape=(jump_data_length, num_columns, 1), name="first_input")
    x = Conv2D(32, kernel_size=(kernel, kernel), padding="same", activation=act_func)(first_input)
    for i in range(c):
        x = Conv2D(32 * (i + 1), kernel_size=(kernel, kernel), padding="same", activation=act_func)(x)
    x = MaxPooling2D(pool_size=(pool, pool), padding="same")(x)
    x = Flatten()(x)
    for i in range(d):
        x = Dense(32 * d, activation=act_func)(x)
    x = Dense(41, activation='softmax', name="output")(x)

    model = Model(inputs=first_input, outputs=x)
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy'])

    return model


def grid_search_build(x_train, y_train, x_test, y_test, jump_data_length):

    best_score = 0
    best_model = 0

    for c in range(4):
        for kernel in range(2, 5):
            for pool in range(2, 5):
                for d in range(2, 5):
                    for act_func in ['tanh', 'elu']:
                        for loss in ['categorical_crossentropy', 'poisson', 'kl_divergence']:
                            for optim in ['adam', 'Adamax', 'Nadam']:
                                model_parameters = [c, kernel, pool, d, act_func, loss, optim]
                                print("Training model with: " + str(model_parameters))

                                model = build_model(jump_data_length, c, kernel, pool, d, act_func, loss, optim)
                                model.fit(x_train, y_train, batch_size=32, epochs=12, verbose=1)
                                score = model.evaluate(x_test, y_test, verbose=1)
                                if score[1] > best_score:
                                    best_score = score[1]
                                    best_model = model
                                    best_model_parameters = model_parameters
                                    print("Best model with: " + str(best_model_parameters))

    print("The best model is: c = %d, kernel = %d, pool = %d, d = %d, act_func = %s, loss = %s, optim = %s"%(
        best_model_parameters[0], best_model_parameters[1], best_model_parameters[2], best_model_parameters[3],
        best_model_parameters[4], best_model_parameters[5], best_model_parameters[6]))

    return best_model


def run_multiple_times(runs, jump_data_length, num_columns, c, kernel, pool, d, act_func, loss, optim, x_train, y_train, x_test, y_test, epochs):

    best_score = 0
    mean_score = 0

    for i in range(runs):
        model = build_model(jump_data_length, num_columns, c, kernel, pool, d, act_func, loss, optim)
        #model = build_model_testing(jump_data_length, num_columns)
        model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1)
        score = model.evaluate(x_test, y_test, verbose=1)
        mean_score += score[1]
        if score[1] > best_score:
            best_score = score[1]
            best_model = model
    mean_score = mean_score / runs

    print("Best score: %f, Mean score: %f"%(best_score * 100, mean_score * 100))

    return best_model


def main():

    x_train, y_train, x_test, y_test, jump_data_length, num_columns = prepare_data()
    # model = grid_search_build(x_train, y_train, x_test, y_test, jump_data_length)
    # model = build_model_testing(jump_data_length, x_train, y_train)
    # model = run_multiple_times(10, jump_data_length, 3, 3, 2, 2, 'tanh', 'kl_divergence', 'Nadam', x_train, y_train, x_test, y_test, 20)
    model = run_multiple_times(10, jump_data_length, num_columns, 1, 3, 2, 2, 'tanh', 'categorical_crossentropy', 'Nadam', x_train, y_train, x_test, y_test, 25)
    model.evaluate(x_test, y_test, verbose=1)

    return


if __name__ == '__main__':
    main()
