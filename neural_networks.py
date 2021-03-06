import pandas as pd
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Dropout, SpatialDropout2D, AveragePooling2D
from keras import backend as k
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
import shap
import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from holoviews.plotting.util import process_cmap
import holoviews as hv
import pickle


def prepare_data(data_train, data_test, pp_list, only_pp=None):
    """
    Prepares the data for the CNN by reshaping the X... data to a 4-dim ndarray and one-hot-encoding the y... data

    Parameters
    ----------
    data_train : pandas.Dataframe
        dataframe read from .csv file
    data_test : pandas.Dataframe
        dataframe read from .csv file
    pp_list : list
        a list with values from 1 to 4: [1, 2, 3, 4]. Corresponds to the blocks of preprocessed data. 1: first 9 columns, 2, 3, 4: 12 columns each
    only_pp : None or any
        if not None: drops all columns except for the preprocessed data

    :return: x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes
    """
    first_djumps = set([col for col in data_train.columns if 'DJump' in col]) - set(
        [col for col in data_train.columns if 'DJump_SIG_I_S' in col]) \
                   - set([col for col in data_train.columns if 'DJump_ABS_I_S' in col]) - set(
        [col for col in data_train.columns if 'DJump_I_ABS_S' in col])
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

    num_columns = len(data_train.columns) - 2

    jump_data_length = len(data_train[data_train['SprungID'] == data_train['SprungID'].unique()[0]])

    for id in data_train['SprungID'].unique():
        subframe = data_train[data_train['SprungID'] == id]
        y_train.append(subframe['Sprungtyp'].unique()[0])
        # 'Time', 'TimeInJump',
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

    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], jump_data_length, num_columns, 1)
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], jump_data_length, num_columns, 1)

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    num_classes = len(y_train.columns)

    return x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes


def prepare_data_oneliner(data_train, data_test, pp_list, only_pp=None):
    """
    Prepares the data for the DFF and one-hot-encoding of y... data

    Parameters
    ----------
    data_train : pandas.Dataframe
        dataframe read from .csv file
    data_test : pandas.Dataframe
        dataframe read from .csv file
    pp_list : list
        a list with values from 1 to 4: [1, 2, 3, 4]. Corresponds to the blocks of preprocessed data. 1: first 9 columns, 2, 3, 4: 12 columns each
    only_pp : None or any
        if not None: drops all columns except for the preprocessed data

    :return: x_train, y_train, x_test, y_test, num_columns, num_classes
    """
    first_djumps = set([col for col in data_train.columns if 'DJump' in col]) - set(
        [col for col in data_train.columns if 'DJump_SIG_I_S' in col]) \
                   - set([col for col in data_train.columns if 'DJump_ABS_I_S' in col]) - set(
        [col for col in data_train.columns if 'DJump_I_ABS_S' in col])
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


def build_model(jump_data_length, num_columns, num_classes, c, kernel, pool, d, act_func, loss, optim):
    """
    Builds a CNN model.

    Parameters
    ----------
    jump_data_length : int
        how many rows for each jump. Return value of prepare_data_CNN
    num_columns : int
        how many columns exist. Return value of prepare_data_CNN
    num_classes : int
        how many classes exist. Return value of prepare_data_CNN
    conv : int
        how many extra convolutional layers to add. 1 is the base
    kernel : int
        kernel size of convolutional layers.
    pool : int
        pool size of MaxPooling layers.
    dense : int
        how many dense layers to add.
    act_func : str
        activation function for all layers except the last
    loss : str
        loss
    optim : str
        optimizer

    :return: model
    """
    first_input = Input(shape=(jump_data_length, num_columns, 1), name="first_input")
    x = Conv2D(32, kernel_size=(kernel, kernel), padding="same", activation=act_func)(first_input)
    for i in range(c):
        x = Conv2D(32 * (i + 1), kernel_size=(kernel, kernel), padding="same", activation=act_func)(x)
    x = MaxPooling2D(pool_size=(pool, pool), padding="same")(x)
    x = Flatten()(x)
    for i in range(d):
        x = Dense(32 * d, activation=act_func)(x)
    x = Dense(num_classes, activation='softmax', name="output")(x)

    model = Model(inputs=first_input, outputs=x)
    model.compile(loss=loss, optimizer=optim,
                  metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                           keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def build_model_grid(jump_data_length=20, num_columns=16, num_classes=43, c=1, act_func='tanh',
                     loss='categorical_crossentropy', optim='Nadam'):
    """
    Nearly the same as build_model_CNN, just a sequential model. Have to use this for grid search.

    Parameters
    ----------
    jump_data_length : int
        how many rows for each jump. Return value of prepare_data_CNN
    num_columns : int
        how many columns exist. Return value of prepare_data_CNN
    num_classes : int
        how many classes exist. Return value of prepare_data_CNN
    c : int
        how many extra convolutional layers to add. 1 is the base
    act_func : str
        activation function for all layers except the last
    loss : str
        loss
    optim : str
        optimizer

    :return: sequential model
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation=act_func,
                     input_shape=(jump_data_length, num_columns, 1)))
    for i in range(c):
        model.add(Conv2D(32 * (i + 1), kernel_size=(3, 3), padding="same", activation=act_func))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(64, activation=act_func))
    model.add(Dense(num_classes, activation='softmax', name="output"))

    model.compile(loss=loss, optimizer=optim,
                  metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                           keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def run_multiple_times(jump_data_length, num_columns, num_classes, runs, conv, kernel, pool, dense, act_func, loss,
                       optim, epochs, x_train, y_train, x_test, y_test):
    """
    Builds multiple CNN models and chooses the best one according to accuracy

    Parameters
    ----------
    jump_data_length : int
        how many rows for each jump. Return value of prepare_data_CNN
    num_columns : int
        how many columns exist. Return value of prepare_data_CNN
    num_classes : int
        how many classes exist. Return value of prepare_data_CNN
    runs : int
        how many models to train
    conv : int
        how many extra convolutional layers to add. 1 is the base
    kernel : int
        kernel size of convolutional layers.
    pool : int
        pool size of MaxPooling layers.
    dense : int
        how many dense layers to add.
    act_func : str
        activation function for all layers except the last
    loss : str
        loss
    optim : str
        optimizer
    epochs : int
        how many epochs to train each model
    x_train : pandas.Dataframe
        prepared data
    y_train : pandas.Dataframe
        prepared data
    x_test : pandas.Dataframe
        prepared data
    y_test : pandas.Dataframe
        prepared data

    :return: best model of all runs
    """
    best_score = 0
    mean_score = 0

    for i in range(runs):
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
        model = build_model(jump_data_length, num_columns, num_classes, conv, kernel, pool, dense, act_func, loss,
                            optim)
        # model = build_model_testing(jump_data_length, num_columns)
        model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1, callbacks=[callback])
        score = model.evaluate(x_test, y_test, verbose=1)
        mean_score += score[1]
        if score[1] > best_score:
            best_score = score[1]
            best_scores = score
            best_model = model
    mean_score = mean_score / runs

    print("Best score: %f, Mean score: %f" % (best_score * 100, mean_score * 100))

    best_model.summary()

    tp = best_scores[2]
    tn = best_scores[3]
    fp = best_scores[4]
    fn = best_scores[5]
    youden = (tp / (fn + tp) + (tn / (tn + fp))) - 1
    prec = tp / (tp + fp)
    rec = tp / (fn + tp)
    fscore = 2 * (prec * rec) / (prec + rec)
    print("youden: " + str(round(youden, 4)))
    print("fscore: " + str(round(fscore, 4)))

    return best_model


def build_model_oneliner(num_columns, num_classes, act_func, loss, optim):
    """
    Builds a DFF model.

    Parameters
    ----------
    num_columns : int
        how many columns exist. Return value of prepare_data_DFF
    num_classes : int
        how many classes exist. Return value of prepare_data_DFF
    act_func : str
        activation function
    loss : str
        loss
    optim : str
        optimizer

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


def run_multiple_times_oneliner(num_columns, num_classes, runs, act_func, loss, optim, epochs, x_train, y_train, x_test,
                                y_test):
    """
    Builds multiple DFF models and chooses the best one according to accuracy

    Parameters
    ----------
    num_columns : int
        how many columns exist. Return value of prepare_data_DFF
    num_classes : int
        how many classes exist. Return value of prepare_data_DFF
    runs : int
        how many models to train
    act_func : str
        activation function for all layers except the last
    loss : str
        loss
    optim : str
        optimizer
    epochs : int
        how many epochs to train each model
    x_train : pandas.Dataframe
        prepared data
    y_train : pandas.Dataframe
        prepared data
    x_test : pandas.Dataframe
        prepared data
    y_test : pandas.Dataframe
        prepared data

    :return: best model of all runs
    """
    best_score = 0
    mean_score = 0

    for i in range(runs):
        callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=11, restore_best_weights=True)
        model = build_model_oneliner(num_columns, num_classes, act_func, loss, optim)
        model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1, callbacks=[callback])
        score = model.evaluate(x_test, y_test, verbose=1)
        mean_score += score[1]
        if score[1] > best_score:
            best_score = score[1]
            best_scores = score
            best_model = model
    mean_score = mean_score / runs

    print("Best score: %f, Mean score: %f" % (best_score * 100, mean_score * 100))

    best_model.summary()

    tp = best_scores[2]
    tn = best_scores[3]
    fp = best_scores[4]
    fn = best_scores[5]
    youden = (tp / (fn + tp) + (tn / (tn + fp))) - 1
    prec = tp / (tp + fp)
    rec = tp / (fn + tp)
    fscore = 2 * (prec * rec) / (prec + rec)
    print("youden: " + str(round(youden, 4)))
    print("fscore: " + str(round(fscore, 4)))

    return best_model


def sample_x_test(x_test, y_test, num, cnn=False):
    """
    Samples data by retrieving only a certain number of each jump.

    Parameters
    ----------
    :param x_test : pandas.Dataframe
        can be x_test and x_train
    :param y_test : pandas.Dataframe
        can be y_test and y_train
    :param num : int
        number of each jump to retrieve
    :param cnn : bool
        check True for CNN

    :return: sampled data Dataframe
    """
    if cnn:
        y = y_test.idxmax(axis=1)
        counts = y.value_counts()
        counts = counts.where(counts < num, other=num)
        x = pd.DataFrame()

        for jump in y.unique():
            indexes = y[y == jump].index
            subframe = pd.DataFrame()
            for index in indexes:
                subframe = subframe.append({'Sprungtyp': jump, 'data': x_test[index]}, ignore_index=True)
            subframe = subframe.sample(counts[jump], random_state=1)
            x = x.append(subframe)

        x = x.sample(frac=1, random_state=1)  # shuffle
        y = x['Sprungtyp']
        y = y.reset_index(drop=True)
        x = x.drop(['Sprungtyp'], axis=1)
        x = x.reset_index(drop=True)

        shap_x = np.array
        for i in range(len(x)):
            if i == 0:
                shap_x = x.loc[i][0]
                shap_x = shap_x[np.newaxis, ...]
                continue
            row = x.loc[i][0]
            shap_x = np.insert(shap_x, i, row, axis=0)

        return shap_x, y

    else:
        df = x_test.copy()
        df['Sprungtyp'] = y_test.idxmax(axis=1)  # if not one-hot-encoded:   df['Sprungtyp'] = y_test
        counts = df['Sprungtyp'].value_counts()
        counts = counts.where(counts < num, other=num)
        x = pd.DataFrame(columns=df.columns)

        for jump in df['Sprungtyp'].unique():
            subframe = df[df['Sprungtyp'] == jump]
            x = x.append(subframe.sample(counts[jump], random_state=1), ignore_index=True)

        x = x.sample(frac=1, random_state=1)  # shuffle
        y = x['Sprungtyp']
        y = y.reset_index(drop=True)
        x = x.drop(['Sprungtyp'], axis=1)
        for column in x.columns:
            x[column] = x[column].astype(float).round(3)

    return x, y


def get_index(jump_list, y_test):
    """
    Get the indexes for all jumps in the jump_list

    Parameters
    ----------
    jump_list: list
        list with jump names
    y_test: pandas.Dataframe
        y_test

    :return: a list with indexes of all jumps in the jump_list
    """
    index_list = []
    for jump in jump_list:
        index_list.extend(y_test.where(y_test == jump).dropna().index.tolist())
    return index_list


def jump_core_detection(modeltype, data_train, data_test, pp_list, jump_length=0):
    """
    Trains many different models with differently cut data. We cut from back to front, front to back, and from both sides

    Parameters
    ----------
    modeltype : str
        'CNN' or 'DFF'
    data_train : pandas.Dataframe
        dataframe read from .csv file
    data_test : pandas.Dataframe
        dataframe read from .csv file
    pp_list : list
        a list with values from 1 to 4: [1, 2, 3, 4]. Corresponds to the blocks of preprocessed data. 1: first 9 columns, 2, 3, 4: 12 columns each
    jump_length: int
        how many datapoints for jump
    :return: dictionary with scores of all trained models
    """

    if modeltype == 'dff':
        jump_length = int(len(list(data_test.drop([col for col in data_train.columns if 'DJump' in col], axis=1)
                                   .drop(['SprungID', 'Sprungtyp'], axis=1).columns)) \
                          / len(
            [c for c in list(data_test.drop([col for col in data_train.columns if 'DJump' in col], axis=1)
                             .drop(['SprungID', 'Sprungtyp'], axis=1).columns) if c.startswith('0_')]))

    scores = {}
    percentage = int(100 / jump_length)
    full_list = [l for l in range(0, 100, percentage)]

    variants = []
    for i in range(jump_length - 1):
        variants.append([l for l in range(i + 1)])
    for i in range(1, jump_length):
        variants.append(list(range(i, jump_length)))
    for i in range(int((jump_length - 1) / 2)):
        both_sides = list(set(list(range(jump_length))) - set(list(range(0, jump_length))[i + 1:-1 - i]))
        both_sides.sort()
        variants.append(both_sides)

    for variant in variants:  # 0 = first data point    ->     jump_length = last data points

        if modeltype == 'cnn':
            indexes = []
            for i in range(int(len(data_train) / jump_length)):
                for to_delete in variant:
                    indexes.append(i * jump_length + to_delete)
            data_train_copy = data_train.drop(indexes)
            indexes = []
            for i in range(int(len(data_test) / jump_length)):
                for to_delete in variant:
                    indexes.append(i * jump_length + to_delete)
            data_test_copy = data_test.drop(indexes)

            x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes = prepare_data(data_train_copy,
                                                                                                        data_test_copy,
                                                                                                        pp_list)

            model = run_multiple_times(jump_data_length, num_columns, num_classes, runs=2, conv=3, kernel=3, pool=2,
                                       dense=2, act_func='tanh', loss='categorical_crossentropy', optim='Nadam',
                                       epochs=40,
                                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            score = model.evaluate(x_test, y_test, verbose=1)

            variant_output = [v * percentage for v in variant]
            variant_output = list(set(full_list) - set(variant_output))
            variant_output.sort()

            scores[str(variant_output[0]) + ' - ' + str(variant_output[-1])] = score[1]

        elif modeltype == 'dff':
            data_train_copy = data_train.copy()
            data_test_copy = data_test.copy()
            for to_delete in variant:
                data_train_copy = data_train_copy.drop(
                    [c for c in data_train.columns if c.startswith(str(int(to_delete * percentage)) + '_')], axis=1)
                data_test_copy = data_test_copy.drop(
                    [c for c in data_train.columns if c.startswith(str(int(to_delete * percentage)) + '_')], axis=1)

            x_train, y_train, x_test, y_test, num_columns, num_classes = prepare_data_oneliner(data_train_copy,
                                                                                               data_test_copy, pp_list)

            model = run_multiple_times_oneliner(num_columns, num_classes, runs=3, act_func='relu',
                                                loss='categorical_crossentropy', optim='Nadam', epochs=100,
                                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            score = model.evaluate(x_test, y_test, verbose=1)

            variant_output = [v * percentage for v in variant]
            variant_output = list(set(full_list) - set(variant_output))
            variant_output.sort()

            scores[str(variant_output[0]) + ' - ' + str(variant_output[-1])] = score[1]

    print(scores)

    min_y_value = 70
    plt.figure(figsize=(13, 13))
    plt.suptitle('DFF with pp: percentage_mean_std_25')
    plt.xlabel('Data')
    plt.ylabel('Accuracy')
    plt.axis([0, full_list[-1], min_y_value, 100])
    plt.xticks(range(0, 100 + percentage, percentage))
    plt.yticks(range(min_y_value, 105, 5))
    plt.grid(True, axis='x')
    # cmap = process_cmap('brg', len(scores))

    for i in range(len(scores)):
        entry = list(scores.items())[i]
        start, end = entry[0].split('-')
        acc = entry[1] * 100
        if int(acc) >= min_y_value:
            if start.replace(' ', '') == '0':
                plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#0000ff', alpha=0.7)
            elif end.replace(' ', '') == str(full_list[-1]):
                plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#ff0000', alpha=0.7)
            else:
                plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#00ff00', alpha=0.7)

    plt.show()


def create_colormap(shap_y_test, shap_values=None):
    """
    Takes values and creates unique colormap, so that each jump, always has the same color.
    Function is needed in bar_plots for 'single' and 'summary_color' bar plots
    -> contains color dict for all jumps, that are used until now, so must be updated if more jumps available.
    An overview of the colors can be found in cmap.pdf

    Parameters
    ----------
    shap_values : list of arrays
        output from explainer.shap_values(shap_x_test); If shap_values are None, shap_y_test must be a string.
    shap_y_test : pandas.Dataframe
        sampled y_test or single string jump

    :return: colormap or single color if shap_values=None
    """
    color_dict = {
        '1 3/4 Salto vw B': '#843c39',
        '1 3/4 Salto vw C': '#ad494a',
        '3/4 Salto rw A': '#d6616b',
        '3/4 Salto vw A': '#e7969c',
        'Barani A': '#e6550d',
        'Barani B': '#fd8d3c',
        'Barani C': '#fdae6b',
        'Bauchsprung': '#544000',
        'B??cksprung': '#8c6d31',
        'Gr??tschwinkel': '#bd9e39',
        'Hocksprung': '#e7ba52',
        'Von Bauch in Stand': '#ffe07d',
        'Strecksprung': '#e7cb94',
        'Fliffis B': '#7b4173',
        'Fliffis C': '#a55194',
        'Baby- Fliffis C': '#ce6dbd',
        'Fliffis aus B': '#d9599c',
        'Fliffis aus C': '#f781bf',
        'Fliffis- Rudi B': '#ff8cdb',
        'Fliffis- Rudi C': '#ffb3e7',
        'Rudi': '#c91a93',
        'Halb ein Triffis C': '#756bb1',
        'Triffis B': '#9e9ac8',
        'Triffis C': '#bcbddc',
        'Schraubensalto A': '#057d59',
        'Schraubensalto': '#099e72',
        'Schraubensalto C': '#3fc49d',
        'Voll- ein 1 3/4 Salto vw C': '#637939',
        'Voll- ein- Rudi- aus B': '#8ca252',
        'Voll- ein- halb- aus B': '#b5cf6b',
        'Doppelsalto A': '#3182bd',
        'Doppelsalto B': '#6baed6',
        'Doppelsalto C': '#9ecae1',
        'Salto rw A': '#393b79',
        'Salto rw B': '#5254a3',
        'Salto rw C': '#6b6ecf',
        'Salto A': '#24b8bf',
        'Salto B': '#64ded2',
        'Salto C': '#a5e8e8',
        'Voll- ein- voll- aus A': '#31a354',
        'Voll- ein- voll- aus B': '#74c476',
        'Voll- ein- voll- aus C': '#a1d99b',
        '1/2 ein 1/2 aus C': '#636363',
        'Cody C': '#d9d9d9'}

    if shap_values is None:
        cmap = color_dict[shap_y_test]
    else:
        class_sequence = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
        d = dict(enumerate(np.array(shap_y_test.unique()).flatten(), 0))
        index_names = np.vectorize(lambda i: d[i])(class_sequence)

        list_cmap = np.vectorize(lambda i: color_dict[i])(index_names)
        cmap = ListedColormap(list_cmap)

    return cmap


def bar_plots(shap_values, shap_x_test, shap_y_test, bar='summary', size=None, jumps=None, folder=None, name=None, max_display=None):
    """
    Function used to plot different bar charts. Used for global feature importance, global feature importance per class,
    feature importance per class and percentual feature importance across the jumps

    Parameters
    ----------

    shap_values : list of arrays
        output from explainer.shap_values(shap_x_test)
    shap_x_test : pandas.Dataframe
        sampled x data
    shap_y_test : pandas.Dataframe
        sampled y data
    bar : str
        'summary', 'summary_color', 'single' or 'percentual', default: 'summary'; different bar plots
    size : tuple (int, int)
        for individual plot size
    jumps : list
        for 'single' and 'percentual' you can input a list with the jump names, for which a bar plot should be created
    folder : str
        path, where the plots should be saved, when its empty, plots will be shown and not be saved
    name : str
        for saving the plot you can choose an other name
    max_display : int
        for 'summary' and 'summary_color' plots, if its none, than all features will be displayed

    :return:
    """
    if size is None:
        plot_size = {
            'summary': (25, 20),
            'summary_color': (25, 20),
            'single': (20, 20),
            'percentual': (30, 18)
        }
        size = plot_size[bar]
    if name is None:
        name = ''
    else:
        name = '-' + name
    if jumps is None:
        jumps = []
    if max_display is None:
        max_display = len(shap_x_test.columns)

    if bar == 'summary':
        if folder is None:
            shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=size,
                              color=ListedColormap('#616161'), class_names=shap_y_test.unique(),
                              max_display=max_display)
        else:
            shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=size,
                              color=ListedColormap('#616161'), class_names=shap_y_test.unique(),
                              max_display=max_display, show=False)
            plt.savefig(folder + 'summary_plot' + name + '.png')
        plt.clf()
    elif bar == 'summary_color':
        if folder is None:
            shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=size,
                              color=create_colormap(shap_y_test, shap_values), class_names=shap_y_test.unique(),
                              max_display=max_display)
        else:
            shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=size,
                              color=create_colormap(shap_y_test, shap_values), class_names=shap_y_test.unique(),
                              max_display=max_display, show=False)
            plt.savefig(folder + 'summary_plot_color' + name + '.png')
        plt.clf()
    elif bar == 'single':
        if len(jumps) == 0:
            jumps = shap_y_test.unique()
        for jump in jumps:
            color_string = create_colormap(jump)
            shap.summary_plot(shap_values[np.where(shap_y_test.unique() == jump)[0][0]].__abs__(), shap_x_test,
                              plot_type='bar', color=color_string, plot_size=size, show=False)
            if folder is None:
                plt.show()
            else:
                plt.savefig(folder + jump.replace('/', '-') + '.png')
            plt.clf()
    elif bar == 'percentual':
        feature_names = shap_x_test.columns
        jump_dict = {}
        if len(jumps) == 0:
            jumps = shap_y_test.unique().tolist()
            jumps.sort()
        for jump in jumps:
            values = np.abs(shap_values[np.where(shap_y_test.unique() == jump)[0][0]].__abs__()).mean(0)
            sum_values = np.sum(values)
            jump_dict[jump] = [(v / sum_values) * 100 for v in values]

        # for saving the data, when plot will be saved
        if folder is not None:
            with open(folder+'percentual_plot' + name + '.txt', 'w') as f:
                print(jump_dict, file=f)
        # creates the plot
        labels = list(jump_dict.keys())
        data = np.array(list(jump_dict.values()))
        data_cum = data.cumsum(axis=1)
        # rainbow or turbo cmap
        if len([item for item in feature_names if item.startswith('0')]) > 0:
            category_colors = plt.get_cmap('rainbow')(
                np.linspace(0.15, 0.85, len([item for item in feature_names if item.startswith('0')])))
            # how often category_colors have to repeat itself
            number_sub_cm = len([item for item in feature_names if not item.startswith('DJump')]) / \
                            len([item for item in feature_names if item.startswith('0')])
            category_colors = np.tile(category_colors, (int(number_sub_cm), 1))
        else:
            category_colors = np.ndarray(shape=(1, 4))
        count_djumps = len([item for item in feature_names if item.startswith('DJump')])
        # if Djumps in features, than other colormap must be appended
        if count_djumps > 0:

            dj_12_colors = np.array([
                [0.19216, 0.50980, 0.74118, 1.00000],
                [0.19216, 0.63922, 0.32941, 1.00000],
                [0.90196, 0.33333, 0.05098, 1.00000],
                [0.41961, 0.68235, 0.83922, 1.00000],
                [0.45490, 0.76863, 0.46275, 1.00000],
                [0.99216, 0.55294, 0.23529, 1.00000],
                [0.61961, 0.79216, 0.88235, 1.00000],
                [0.63137, 0.85098, 0.60784, 1.00000],
                [0.99216, 0.68235, 0.41961, 1.00000],
                [0.77647, 0.85882, 0.93725, 1.00000],
                [0.78039, 0.91373, 0.75294, 1.00000],
                [0.99216, 0.81569, 0.63529, 1.00000]])
            dj_9_colors = np.array([
                [0.48235, 0.25490, 0.45098, 1.00000],
                [0.64706, 0.31765, 0.58039, 1.00000],
                [0.80784, 0.42745, 0.74118, 1.00000],
                [0.45882, 0.41961, 0.69412, 1.00000],
                [0.61961, 0.60392, 0.78431, 1.00000],
                [0.73725, 0.74118, 0.86275, 1.00000],
                [0.88235, 0.07843, 0.50588, 1.00000],
                [0.96863, 0.50588, 0.74902, 1.00000],
                [0.96863, 0.73333, 0.82353, 1.00000]])

            # if nine djump pack not included
            if (count_djumps % 12) == 0:
                # if Djump features first, colormap must be appended on the first position
                if feature_names[0].startswith('DJump'):
                    category_colors = np.concatenate((
                        np.tile(dj_12_colors, (int(count_djumps/12), 1)),
                        category_colors))
                else:
                    category_colors = np.concatenate((
                        category_colors,
                        np.tile(dj_12_colors, (int(count_djumps/12), 1))))
            # if djump contains nine pack
            else:
                c = int((count_djumps - 9) / 12)
                sub_cm = np.concatenate((dj_9_colors, np.tile(dj_12_colors, (c, 1))))
                if feature_names[0].startswith('DJump'):
                    category_colors = np.concatenate((sub_cm, category_colors))
                else:
                    category_colors = np.concatenate((category_colors, sub_cm))

        fig, ax = plt.subplots(figsize=size)
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())
        for i, (colname, color) in enumerate(zip(feature_names.tolist(), category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.6,
                            label=colname, color=color)

            r, g, b, _ = color
            # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # ax.bar_label(rects, label_type='center', color=text_color)
            if len(feature_names) < 70:
                ax.legend(ncol=1, bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0.)
            else:
                ax.legend(ncol=2, bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0.)
        plt.suptitle('Percentual impact bar plot of the individual features')

        if folder is None:
            plt.show()
        else:
            plt.savefig(folder + 'percentual_plot' + name + '.png')
        plt.clf()
    return


def main():
    neural_network = 'dff'  # 'dff'  'cnn'
    run_modus = 'load'  # 'multi' 'grid' 'core' or 'load'
    run = 100  # for multi runs or how often random grid search runs
    data = 'without' # or 'without' or 'only'
    percentage = '20'
    dataset = 'percentage_mean_std'
    pp_list = [3]
    saving = False #or False

    if neural_network == 'cnn':
        data_train = pd.read_csv(
            "Sprungdaten_processed/" + data + "_preprocessed/percentage/" + percentage + "/" + dataset + "_" + percentage +"_train.csv")
        data_test = pd.read_csv(
            "Sprungdaten_processed/" + data + "_preprocessed/percentage/" + percentage + "/" + dataset + "_" + percentage +"_test.csv")
        x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes = prepare_data(data_train,
                                                                                                    data_test, pp_list)
        if run_modus == 'multi':
            model = run_multiple_times(jump_data_length, num_columns, num_classes, runs=run, conv=3, kernel=3, pool=2,
                                       dense=2,
                                       act_func='tanh', loss='categorical_crossentropy', optim='Nadam', epochs=40,
                                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            model.evaluate(x_test, y_test, verbose=1)
        if run_modus == 'grid':
            param_grid = {'num_columns': [num_columns], 'epochs': [30, 40, 50],
                          'conv': [1, 2, 3],
                          'batch_size': [32], 'optim': ['adam', 'Nadam'],
                          'act_func': ['tanh', 'relu', 'sigmoid'],
                          'loss': ['categorical_crossentropy', 'kl_divergence']}
            model = KerasClassifier(build_fn=build_model_grid, verbose=0)
            grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, verbose=1, n_iter=run, n_jobs=1)
            grid_result = grid.fit(x_test, y_test)
            print(grid_result.best_params_)
        if run_modus == 'core':
            jump_core_detection('cnn', data_train, data_test, pp_list, jump_data_length)

        if run_modus == 'load':
            if data == 'with':
                model = keras.models.load_model("models/CNN_with_mean_std_20")
            if data == 'without':
                model = keras.models.load_model("models/CNN_without_mean_5")

    if neural_network == 'dff':
        data_train = pd.read_csv(
            "Sprungdaten_processed/" + data + "_preprocessed/percentage/" + percentage + "/vector_" + dataset + "_" + percentage + "_train.csv")
        data_test = pd.read_csv(
            "Sprungdaten_processed/" + data + "_preprocessed/percentage/" + percentage + "/vector_" + dataset + "_" + percentage + "_test.csv")
        x_train, y_train, x_test, y_test, num_columns, num_classes = prepare_data_oneliner(data_train, data_test,
                                                                                           pp_list)
        if run_modus == 'multi':
            model = run_multiple_times_oneliner(num_columns, num_classes, runs=run, act_func='relu',
                                                loss='categorical_crossentropy',
                                                optim='adam', epochs=40, x_train=x_train, y_train=y_train,
                                                x_test=x_test, y_test=y_test)
            model.evaluate(x_test, y_test, verbose=1)
        if run_modus == 'grid':
            param_grid = {'num_columns': [num_columns], 'epochs': [30, 40, 50, 60, 80],
                          'batch_size': [32], 'optim': ['adam', 'Nadam'],
                          'act_func': ['tanh', 'relu', 'sigmoid'],
                          'loss': ['categorical_crossentropy', 'kl_divergence']}
            model = KerasClassifier(build_fn=build_model_oneliner, verbose=0)
            grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, verbose=1, n_iter=run, n_jobs=1)
            grid_result = grid.fit(x_test, y_test)
            print(grid_result.best_params_)
        if run_modus == 'core':
            jump_core_detection('dff', data_train, data_test, pp_list)
        if run_modus == 'load':
            if data == 'with':
                model = keras.models.load_model("models/DFF_with_mean_std_25")
            if data == 'without':
                model = keras.models.load_model("models/DFF_without_mean_std_20")

    # for saving model
    if saving:
        keras.models.save_model(model, "models/" + neural_network + "_" + data + "_" + dataset + "_" + percentage)

    # model.summary()

    shap.initjs()

    # colormap
    cmap = ['#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31',
            '#bd9e39', '#e7ba52',
            '#e7cb94', '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6',
            '#3182bd', '#6baed6',
            '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476', '#a1d99b',
            '#c7e9c0', '#756bb1',
            '#9e9ac8', '#bcbddc', '#dadaeb', '#636363', '#969696', '#969696', '#d9d9d9', '#f0027f', '#f781bf',
            '#f7b6d2', '#fccde5',
            '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9',
            '#bc80bd', '#ccebc5', '#ffed6f']

    cmap_cm = process_cmap('summer')
    cmap_cm.insert(0, '#ffffff')
    cmap_cm.insert(-1, '#000000')
    cmap_cm = ListedColormap(cmap_cm)

    cmap_cm_AJ = ['#ffffff', '#048166']
    cmap_cm_AJ = ListedColormap(cmap_cm_AJ)

    if neural_network == 'dff':
    # DFF
        if run_modus == 'load':
            with open('plots/DFF/' + data + '_preprocessed/shap_data.pkl', 'rb') as f:
                shap_values, shap_x_train, shap_y_train, shap_x_test, shap_y_test = pickle.load(f)
        else:
            shap_x_test, shap_y_test = sample_x_test(x_test, y_test, 3)
            shap_x_train, shap_y_train = sample_x_test(x_train, y_train, 6)
            # background = shap.sample(shap_x_train, 400, random_state=1)
            explainer = shap.KernelExplainer(model, shap_x_train)
            shap_values = explainer.shap_values(shap_x_test)

        bar_cm = create_colormap(shap_y_test, shap_values)

        if saving:

            with open('plots/DFF/' + data + '_preprocessed/' + 'shap_data_' + dataset + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([shap_values, shap_x_train, shap_y_train, shap_x_test, shap_y_test], f)

        # Beeswarm plots
        saltoA = np.where(shap_y_test.unique() == 'Salto A')[0][0]
        shap.summary_plot(shap_values[saltoA], shap_x_test, plot_size=(25, 15), title='Salto A')
        saltoB = np.where(shap_y_test.unique() == 'Salto B')[0][0]
        shap.summary_plot(shap_values[saltoB], shap_x_test, plot_size=(25, 15), title='Salto B')
        saltoC = np.where(shap_y_test.unique() == 'Salto C')[0][0]
        shap.summary_plot(shap_values[saltoC], shap_x_test, plot_size=(25, 15), title='Salto C')

        # percentage plots
        bar_plots(shap_values, shap_x_test, shap_y_test, bar='percentual')

        bar_plots(shap_values, shap_x_test, shap_y_test, bar='percentual',
                  jumps=['Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C', 'Schraubensalto',
                         'Schraubensalto A', 'Schraubensalto C', 'Doppelsalto B', 'Doppelsalto C'], name='Saltos')

    if neural_network == 'cnn':

        shap_x_test, shap_y_test = sample_x_test(x_test, y_test, 3, cnn=True)
        shap_x_train, shap_y_train = sample_x_test(x_train, y_train, 6, cnn=True)

        parts = {1: ['1 3/4 Salto vw B', '1 3/4 Salto vw C', '1/2 ein 1/2 aus C', '3/4 Salto rw A', '3/4 Salto vw A',
                     'Baby- Fliffis C'],
                 2: ['Barani A', 'Barani B', 'Barani C', 'Cody C', 'Rudi'],
                 3: ['Bauchsprung', 'B??cksprung', 'Gr??tschwinkel', 'Hocksprung', 'Von Bauch in Stand', 'Strecksprung'],
                 4: ['Fliffis B', 'Fliffis C', 'Fliffis aus B', 'Fliffis aus C', 'Fliffis- Rudi B', 'Fliffis- Rudi C'],
                 5: ['Halb ein Triffis C', 'Triffis B', 'Triffis C'],
                 6: ['Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C'],
                 7: ['Schraubensalto', 'Schraubensalto A', 'Schraubensalto C', 'Doppelsalto B', 'Doppelsalto C'],
                 8: ['Voll- ein 1 3/4 Salto vw C', 'Voll- ein- Rudi- aus B', 'Voll- ein- halb- aus B',
                     'Voll- ein- voll- aus A', 'Voll- ein- voll- aus B', 'Voll- ein- voll- aus C']}

        for j in [1, 2, 3, 4, 5, 6, 7, 8]:
            index_list = get_index(parts[j], shap_y_test)
            to_explain = shap_x_test[index_list]
            explainer = shap.DeepExplainer(model, shap_x_train)
            shap_values, indexes = explainer.shap_values(to_explain, ranked_outputs=4, check_additivity=False)
            d = dict(enumerate(np.array(y_test.columns).flatten(), 0))
            index_names = np.vectorize(lambda i: d[i])(indexes)

            with open('plots/CNN/without_preprocessed/shap_data' + str(j) + '.pkl', 'wb') as f:
                pickle.dump([shap_values, to_explain, index_names], f)

            shap.image_plot(shap_values, to_explain, index_names, show=False)
            # plt.savefig('plots/CNN/with_preprocessed/CNN_with_mean_std_20_part' + str(j) + '.png')
        """
        # Shap for specific Class
        i = y_test.index[y_test['Salto C'] == 1]
        pd.DataFrame(model.predict(x_test[i]), columns=y_test.columns).idxmax(axis=1)
        background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(x_test[i])
        shap.image_plot(shap_values, -x_test[i])  # , labels=list(y_test.columns))
        """

    # Confusion matrix to find mistakes in classification
    cm = sklearn.metrics.confusion_matrix(y_test.idxmax(axis=1),
                                          pd.DataFrame(model.predict(x_test), columns=y_test.columns).idxmax(
                                              axis=1))
    if saving:
        # save data:
        if neural_network == 'dff':
            pd.DataFrame(cm, columns=y_test.columns, index=y_test.columns).to_csv(
                'plots/DFF/' + data + '_preprocessed/confusion_matrix_' + dataset + '.csv')
        if neural_network == 'cnn':
            pd.DataFrame(cm, columns=y_test.columns, index=y_test.columns).to_csv(
                'plots/CNN/' + data + '_preprocessed/confusion_matrix_' + dataset + '.csv')

    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_test.columns)
    disp.plot(cmap=cmap_cm)
    disp.figure_.set_figwidth(35)
    disp.figure_.set_figheight(25)
    disp.figure_.autofmt_xdate()
    if saving:
        if neural_network == 'dff':
            plt.savefig('plots/DFF/DFF_confusion_matrix_' + dataset + '.png')
        if neural_network == 'cnn':
            plt.savefig('plots/CNN/CNN_confusion_matrix_' + dataset + '.png')

    else:
        plt.show()

    return


if __name__ == '__main__':
    main()
