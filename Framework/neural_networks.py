import pandas as pd
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


"""
#                                       #
#                                       #
#               Both NN                 #
#                                       #
#                                       #
"""


def calc_scores(scores):
    """
    Prints the youden and the fscore for a model

    :param scores: the return value of model.evaluate, contains accuracy, tp, tn, fp, fn
    """

    tp = scores[2]
    tn = scores[3]
    fp = scores[4]
    fn = scores[5]
    youden = (tp / (fn + tp) + (tn / (tn + fp))) - 1
    prec = tp / (tp + fp)
    rec = tp / (fn + tp)
    fscore = 2 * (prec * rec) / (prec + rec)
    print("youden: " + str(round(youden, 4)))
    print("fscore: " + str(round(fscore, 4)))


def grid_search(modeltype, param_grid, run, x_test, y_test):
    """
    Uses Randomized Grid Search to find good parameters for the model

    :param modeltype: 'CNN' or 'DFF'
    :param param_grid: all parameters needed for the model
    :param run: how many iterations to be randomly searched
    :return: locally best parameters for modle
    """

    """
    CNN: param_grid = {'num_columns': [num_columns], 'epochs': [30, 40, 50],
                  'conv': [1, 2, 3],
                  'batch_size': [32], 'optim': ['adam', 'Nadam'],
                  'act_func': ['tanh', 'relu', 'sigmoid'],
                  'loss': ['categorical_crossentropy', 'kl_divergence']}

    DFF: param_grid = {'num_columns': [num_columns], 'epochs': [30, 40, 50, 60, 80],
                          'batch_size': [32], 'optim': ['adam', 'Nadam'],
                          'act_func': ['tanh', 'relu', 'sigmoid'],
                          'loss': ['categorical_crossentropy', 'kl_divergence']}
    """

    assert modeltype == 'CNN' or modeltype == 'DFF', "modeltyp not CNN or DFF"

    if modeltype == 'CNN':
        model = KerasClassifier(build_fn=build_model_CNN_sequentiel, verbose=0)
    elif modeltype == 'DFF':
        model = KerasClassifier(build_fn=build_model_DFF, verbose=0)

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, verbose=1, n_iter=run, n_jobs=1)
    grid_result = grid.fit(x_test, y_test)
    print(grid_result.best_params_)


"""
#                                       #
#                                       #
#           Convolutional NN            #
#                                       #
#                                       #
"""


def prepare_data_CNN(data_train, data_test, pp_list, only_pp=None):
    """
    Prepares the data for the CNN by reshaping the X... data to a 4-dim ndarray and one-hot-encoding the y... data

    :param data_train: dataframe read from .csv file
    :param data_test: dataframe read from .csv file
    :param pp_list: a list with values from 1 to 4: [1, 2, 3, 4]. Corresponds to the blocks of preprocessed data. 1: first 9 columns, 2, 3, 4: 12 columns each
    :param only_pp: if not None: drops all columns except for the preprocessed data
    :return: x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes
    """

    first_djumps = set([col for col in data_train.columns if 'DJump' in col]) \
                   - set([col for col in data_train.columns if 'DJump_SIG_I_S' in col]) \
                   - set([col for col in data_train.columns if 'DJump_ABS_I_S' in col]) \
                   - set([col for col in data_train.columns if 'DJump_I_ABS_S' in col])
    if 1 not in pp_list:
        data_train = data_train.drop(first_djumps, axis=1)
        data_test = data_test.drop(first_djumps, axis=1)
    if 2 not in pp_list:
        data_train = data_train.drop([col for col in data_train.columns if 'DJump_SIG_I_S' in col], axis=1)
        data_test = data_test.drop([col for col in data_test.columns if 'DJump_SIG_I_S' in col], axis=1)
    if 3 not in pp_list:
        data_train = data_train.drop([col for col in data_train.columns if 'DJump_ABS_I_S' in col], axis=1)
        data_test = data_test.drop([col for col in data_test.columns if 'DJump_ABS_I_S' in col], axis=1)
    if 4 not in pp_list:
        data_train = data_train.drop([col for col in data_train.columns if 'DJump_I_ABS_S' in col], axis=1)
        data_test = data_test.drop([col for col in data_test.columns if 'DJump_I_ABS_S' in col], axis=1)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for id in data_train['SprungID'].unique():
        subframe = data_train[data_train['SprungID'] == id]
        y_train.append(subframe['Sprungtyp'].unique()[0])
        subframe = subframe.drop(['SprungID', 'Sprungtyp'], axis=1)
        if only_pp is not None:
            subframe = subframe.drop([col for col in subframe.columns if 'DJump' not in col], axis=1)
        x_train.append(subframe)
        print("Preparing train data: " + str(len(x_train)))

    for id in data_test['SprungID'].unique():
        subframe = data_test[data_test['SprungID'] == id]
        y_test.append(subframe['Sprungtyp'].unique()[0])
        subframe = subframe.drop(['SprungID', 'Sprungtyp'], axis=1)
        if only_pp is not None:
            subframe = subframe.drop([col for col in subframe.columns if 'DJump' not in col], axis=1)
        x_test.append(subframe)
        print("Preparing test data: " + str(len(x_test)))

    num_columns = len(data_train.columns) - 2       # without Sprungtyp and SprungID
    jump_data_length = len(data_train[data_train['SprungID'] == data_train['SprungID'].unique()[0]])



    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], jump_data_length, num_columns, 1)
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], jump_data_length, num_columns, 1)

    y_train = pd.get_dummies(y_train)       # one hot encoding
    y_test = pd.get_dummies(y_test)

    num_classes = len(y_train.columns)

    return x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes


def build_model_CNN(jump_data_length, num_columns, num_classes, conv, kernel, pool, dense, act_func, loss, optim):
    """
    Builds a CNN model.

    :param jump_data_length: how many rows for each jump. Return value of prepare_data_CNN
    :param num_columns: how many columns exist. Return value of prepare_data_CNN
    :param num_classes: how many classes exist. Return value of prepare_data_CNN
    :param conv: how many extra convolutional layers to add. 1 is the base
    :param kernel: kernel size of convolutional layers.
    :param pool: pool size of MaxPooling layers.
    :param dense: how many dense layers to add.
    :param act_func: activation function for all layers except the last
    :param loss: loss
    :param optim: optimizer
    :return: model
    """

    first_input = Input(shape=(jump_data_length, num_columns, 1), name="first_input")
    x = Conv2D(32, kernel_size=(kernel, kernel), padding="same", activation=act_func)(first_input)
    for i in range(conv):
        x = Conv2D(32 * (i + 1), kernel_size=(kernel, kernel), padding="same", activation=act_func)(x)
    x = MaxPooling2D(pool_size=(pool, pool), padding="same")(x)
    x = Flatten()(x)
    for i in range(dense):
        x = Dense(32 * dense, activation=act_func)(x)
    x = Dense(num_classes, activation='softmax', name="output")(x)

    model = Model(inputs=first_input, outputs=x)
    model.compile(loss=loss, optimizer=optim,
                  metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                           keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def build_model_CNN_sequentiel(jump_data_length, num_columns, num_classes, conv, act_func, loss, optim):
    """
    Nearly the same as build_model_CNN, just a sequential model. Have to use this for grid search.

    :param jump_data_length: how many rows for each jump. Return value of prepare_data_CNN
    :param num_columns: how many columns exist. Return value of prepare_data_CNN
    :param num_classes: how many classes exist. Return value of prepare_data_CNN
    :param conv: how many extra convolutional layers to add. 1 is the base
    :param act_func: activation function for all layers except the last
    :param loss: loss
    :param optim: optimizer
    :return: sequential model
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation=act_func,
                     input_shape=(jump_data_length, num_columns, 1)))
    for i in range(conv):
        model.add(Conv2D(32 * (i + 1), kernel_size=(3, 3), padding="same", activation=act_func))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(64, activation=act_func))
    model.add(Dense(num_classes, activation='softmax', name="output"))

    model.compile(loss=loss, optimizer=optim,
                  metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                           keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def run_multiple_times_CNN(jump_data_length, num_columns, num_classes, runs, conv, kernel, pool, dense, act_func, loss,
                       optim, epochs, x_train, y_train, x_test, y_test):

    """
    Builds multiple CNN models and chooses the best one according to accuracy

    :param jump_data_length: how many rows for each jump. Return value of prepare_data_CNN
    :param num_columns: how many columns exist. Return value of prepare_data_CNN
    :param num_classes: how many classes exist. Return value of prepare_data_CNN
    :param runs: how many models to train
    :param conv: how many extra convolutional layers to add. 1 is the base
    :param kernel: kernel size of convolutional layers.
    :param pool: pool size of MaxPooling layers.
    :param dense: how many dense layers to add.
    :param act_func: activation function for all layers except the last
    :param loss: loss
    :param optim: optimizer
    :param epochs: how many epochs to train each model
    :param x_train: prepared data
    :param y_train: prepared data
    :param x_test: prepared data
    :param y_test: prepared data
    :return: best model of all runs
    """

    best_score = 0
    mean_score = 0

    for i in range(runs):
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
        model = build_model_CNN(jump_data_length, num_columns, num_classes, conv, kernel, pool, dense, act_func, loss, optim)
        model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1, callbacks=[callback])
        score = model.evaluate(x_test, y_test, verbose=1)
        mean_score += score[1]
        if score[1] > best_score:
            best_score = score[1]
            best_scores = score
            best_model = model
    mean_score = mean_score / runs

    best_model.summary()
    print("Best score: %f, Mean score: %f" % (best_score * 100, mean_score * 100))

    calc_scores(best_scores)

    return best_model



"""
#                                       #
#                                       #
#           Deep Feed Forward           #
#                                       #
#                                       #
"""


def prepare_data_DFF(data_train, data_test, pp_list, only_pp=None):
    """
    Prepares the data for the DFF and one-hot-encoding of y... data

    :param data_train: dataframe read from .csv file
    :param data_test: dataframe read from .csv file
    :param pp_list: a list with values from 1 to 4: [1, 2, 3, 4]. Corresponds to the blocks of preprocessed data. 1: first 9 columns, 2, 3, 4: 12 columns each
    :param only_pp: if not None: drops all columns except for the preprocessed data
    :return: x_train, y_train, x_test, y_test, num_columns, num_classes
    """

    first_djumps = set([col for col in data_train.columns if 'DJump' in col]) \
                   - set([col for col in data_train.columns if 'DJump_SIG_I_S' in col]) \
                   - set([col for col in data_train.columns if 'DJump_ABS_I_S' in col]) \
                   - set([col for col in data_train.columns if 'DJump_I_ABS_S' in col])
    if 1 not in pp_list:
        data_train = data_train.drop(first_djumps, axis=1)
        data_test = data_test.drop(first_djumps, axis=1)
    if 2 not in pp_list:
        data_train = data_train.drop([col for col in data_train.columns if 'DJump_SIG_I_S' in col], axis=1)
        data_test = data_test.drop([col for col in data_test.columns if 'DJump_SIG_I_S' in col], axis=1)
    if 3 not in pp_list:
        data_train = data_train.drop([col for col in data_train.columns if 'DJump_ABS_I_S' in col], axis=1)
        data_test = data_test.drop([col for col in data_test.columns if 'DJump_ABS_I_S' in col], axis=1)
    if 4 not in pp_list:
        data_train = data_train.drop([col for col in data_train.columns if 'DJump_I_ABS_S' in col], axis=1)
        data_test = data_test.drop([col for col in data_test.columns if 'DJump_I_ABS_S' in col], axis=1)

    x_train = data_train.drop('Sprungtyp', axis=1)
    x_train = x_train.drop(['SprungID'], axis=1)
    x_test = data_test.drop('Sprungtyp', axis=1)
    x_test = x_test.drop(['SprungID'], axis=1)

    if only_pp is not None:
        x_train = x_train.drop([col for col in x_train.columns if 'DJump' not in col], axis=1)
        x_test = x_test.drop([col for col in x_test.columns if 'DJump' not in col], axis=1)

    num_columns = len(x_train.columns)

    y_train = data_train['Sprungtyp']
    y_test = data_test['Sprungtyp']

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    num_classes = len(y_train.columns)

    return x_train, y_train, x_test, y_test, num_columns, num_classes


def build_model_DFF(num_columns, num_classes, act_func, loss, optim):
    """
    Builds a DFF model.

    :param num_columns: how many columns exist. Return value of prepare_data_DFF
    :param num_classes: how many classes exist. Return value of prepare_data_DFF
    :param act_func: activation function
    :param loss: loss
    :param optim: optimizer
    :return: model
    """

    first_input = Input(shape=(num_columns,), name="first_input")
    x = Dense(128, activation=act_func)(first_input)
    x = Dense(256, activation=act_func)(x)
    x = Dense(512, activation=act_func)(x)
    x = Dense(512, activation=act_func)(x)
    x = Dense(512, activation=act_func)(x)
    x = Dense(256, activation=act_func)(x)
    x = Dense(128, activation=act_func)(x)
    x = Dense(num_classes, activation='softmax', name="output")(x)

    model = Model(inputs=first_input, outputs=x)
    model.compile(loss=loss, optimizer=optim,
                  metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                           keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def run_multiple_times_DFF(num_columns, num_classes, runs, act_func, loss, optim, epochs, x_train, y_train, x_test, y_test):
    """
    Builds multiple DFF models and chooses the best one according to accuracy

    :param num_columns: how many columns exist. Return value of prepare_data_DFF
    :param num_classes: how many classes exist. Return value of prepare_data_DFF
    :param runs: how many models to train
    :param act_func: activation function for all layers except the last
    :param loss: loss
    :param optim: optimizer
    :param epochs: how many epochs to train each model
    :param x_train: prepared data
    :param y_train: prepared data
    :param x_test: prepared data
    :param y_test: prepared data
    :return: best model of all runs
    """

    best_score = 0
    mean_score = 0

    for i in range(runs):
        callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=11, restore_best_weights=True)
        model = build_model_DFF(num_columns, num_classes, act_func, loss, optim)
        model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1, callbacks=[callback])
        score = model.evaluate(x_test, y_test, verbose=1)
        mean_score += score[1]
        if score[1] > best_score:
            best_score = score[1]
            best_scores = score
            best_model = model
    mean_score = mean_score / runs

    best_model.summary()
    print("Best score: %f, Mean score: %f" % (best_score * 100, mean_score * 100))

    calc_scores(best_scores)

    return best_model
