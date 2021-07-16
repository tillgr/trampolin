import pandas as pd
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import shap
import pickle


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

    Parameters
    ----------
    scores : tuple
        the return value of model.evaluate, contains accuracy, tp, tn, fp, fn

    :return:
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

    Parameters
    ----------
    modeltype : str
        'CNN' or 'DFF'
    param_grid : list
        all parameters needed for the model
    run : int
        how many iterations to be randomly searched
    x_test : pandas.Dataframe
        test data
    y_test : pandas.Dataframe
        classes of test data

    :return: prints locally best parameters for model
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


def jump_core_detection(modeltype, data_train, data_test, pp_list, runs, params):
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
    runs : int
        how many models should be trained for each dataset
    params : list
        a list with all parameters for the model. e.g. CNN: [3, 3, 2, 2, 'tanh', 'categorical_crossentropy', 'Nadam', 40], e.g. DFF: ['relu', 'categorical_crossentropy', 'Nadam', 100]

    :return: dictionary with scores of all trained models
    """

    if modeltype == 'DFF':
        jump_length = int(len(list(data_test.drop([col for col in data_train.columns if 'DJump' in col], axis=1)
                                   .drop(['SprungID', 'Sprungtyp'], axis=1).columns))
                          / len([c for c in list(data_test.drop([col for col in data_train.columns if 'DJump' in col], axis=1)
                                   .drop(['SprungID', 'Sprungtyp'], axis=1).columns) if c.startswith('0_')]))
    elif modeltype == 'CNN':
        x_train, y_train, x_test, y_test, jump_length, num_columns, num_classes = prepare_data_CNN(data_train, data_test, pp_list)
    else:
        raise AttributeError("modeltype is not 'CNN' or 'DFF'")

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

        if modeltype == 'CNN':
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

            x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes = prepare_data_CNN(data_train_copy, data_test_copy, pp_list)

            model = run_multiple_times_CNN(jump_data_length, num_columns, num_classes, runs=runs, conv=params[0], kernel=params[1], pool=params[2],
                                       dense=params[3], act_func=params[4], loss=params[5], optim=params[6], epochs=params[7],
                                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            score = model.evaluate(x_test, y_test, verbose=1)

            variant_output = [v * percentage for v in variant]
            variant_output = list(set(full_list) - set(variant_output))
            variant_output.sort()

            scores[str(variant_output[0]) + ' - ' + str(variant_output[-1])] = score[1]

        elif modeltype == 'DFF':
            data_train_copy = data_train.copy()
            data_test_copy = data_test.copy()
            for to_delete in variant:
                data_train_copy = data_train_copy.drop(
                    [c for c in data_train.columns if c.startswith(str(int(to_delete * percentage)) + '_')], axis=1)
                data_test_copy = data_test_copy.drop(
                    [c for c in data_train.columns if c.startswith(str(int(to_delete * percentage)) + '_')], axis=1)

            x_train, y_train, x_test, y_test, num_columns, num_classes = prepare_data_DFF(data_train_copy, data_test_copy, pp_list)

            model = run_multiple_times_DFF(num_columns, num_classes, runs=runs, act_func=params[0],
                                                loss=params[1], optim=params[2], epochs=params[3],
                                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            score = model.evaluate(x_test, y_test, verbose=1)

            variant_output = [v * percentage for v in variant]
            variant_output = list(set(full_list) - set(variant_output))
            variant_output.sort()

            scores[str(variant_output[0]) + ' - ' + str(variant_output[-1])] = score[1]

    print(scores)

    return scores


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
        if only_pp is None:
            subframe = subframe.drop([col for col in subframe.columns if 'DJump' not in col], axis=1)
        x_train.append(subframe)
        print("Preparing train data: " + str(len(x_train)))

    for id in data_test['SprungID'].unique():
        subframe = data_test[data_test['SprungID'] == id]
        y_test.append(subframe['Sprungtyp'].unique()[0])
        subframe = subframe.drop(['SprungID', 'Sprungtyp'], axis=1)
        if only_pp is None:
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


def gen_shap_CNN(model, part, sample_train, sample_test, x_train, y_train, x_test, y_test):
    """
    generates the shap values for the CNN. We do this in predefined parts to lower the amount of images in the image plot

    Parameters
    ----------
    model : keras.model
        CNN model
    part : int
        1 - 8. related jumps. see code
    sample_train : int
        how often each jump should be sampled into the train data. We used 6
    sample_test : int
        how often each jump should be sampled into the test data. We used 3
    x_train : pandas.Dataframe
        return value of prepare_data_CNN
    y_train : pandas.Dataframe
        return value of prepare_data_CNN
    x_test : pandas.Dataframe
        return value of prepare_data_CNN
    y_test : pandas.Dataframe
        return value of prepare_data_CNN

    :return: shap_values, to_explain and index_names for image plot
    """

    shap_x_train, shap_y_train = sample_x_test(x_train, y_train, sample_train, cnn=True)
    shap_x_test, shap_y_test = sample_x_test(x_test, y_test, sample_test, cnn=True)

    parts = {1: ['1 3/4 Salto vw B', '1 3/4 Salto vw C', '1/2 ein 1/2 aus C', '3/4 Salto rw A', '3/4 Salto vw A', 'Baby- Fliffis C'],
             2: ['Barani A', 'Barani B', 'Barani C', 'Cody C', 'Rudi'],
             3: ['Bauchsprung', 'Bücksprung', 'Grätschwinkel', 'Hocksprung', 'Von Bauch in Stand', 'Strecksprung'],
             4: ['Fliffis B', 'Fliffis C', 'Fliffis aus B', 'Fliffis aus C', 'Fliffis- Rudi B', 'Fliffis- Rudi C'],
             5: ['Halb ein Triffis C', 'Triffis B', 'Triffis C'],
             6: ['Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C'],
             7: ['Schraubensalto', 'Schraubensalto A', 'Schraubensalto C', 'Doppelsalto B', 'Doppelsalto C'],
             8: ['Voll- ein 1 3/4 Salto vw C', 'Voll- ein- Rudi- aus B', 'Voll- ein- halb- aus B', 'Voll- ein- voll- aus A', 'Voll- ein- voll- aus B', 'Voll- ein- voll- aus C']}

    index_list = get_index(parts[part], shap_y_test)
    to_explain = shap_x_test[index_list]

    explainer = shap.DeepExplainer(model, shap_x_train)
    shap_values, indexes = explainer.shap_values(to_explain, ranked_outputs=4, check_additivity=False)

    d = dict(enumerate(np.array(y_test.columns).flatten(), 0))
    index_names = np.vectorize(lambda i: d[i])(indexes)

    with open("plots/CNN/shp_values.pkl", 'wb') as f:
        pickle.dump([shap_values, to_explain, index_names], f)

    return shap_values, to_explain, index_names


def predict_CNN(model, data, pp_list):
    """
    Predicts the classes of given x data

    Parameters
    ----------
    model : keras.model
        a CNN model
    data : pandas.Dataframe
        data to be predicted

    :return: prediction
    """

    x = []

    first_djumps = set([col for col in data.columns if 'DJump' in col]) \
                   - set([col for col in data.columns if 'DJump_SIG_I_S' in col]) \
                   - set([col for col in data.columns if 'DJump_ABS_I_S' in col]) \
                   - set([col for col in data.columns if 'DJump_I_ABS_S' in col])
    if 1 not in pp_list:
        data = data.drop(first_djumps, axis=1)
    if 2 not in pp_list:
        data = data.drop([col for col in data.columns if 'DJump_SIG_I_S' in col], axis=1)
    if 3 not in pp_list:
        data = data.drop([col for col in data.columns if 'DJump_ABS_I_S' in col], axis=1)
    if 4 not in pp_list:
        data = data.drop([col for col in data.columns if 'DJump_I_ABS_S' in col], axis=1)

    for id in data['SprungID'].unique():
        subframe = data[data['SprungID'] == id]
        subframe = subframe.drop(['SprungID'], axis=1)
        num_columns = len(subframe.columns)
        jump_data_length = len(subframe)
        x.append(subframe)
        print("Preparing data for prediction: " + str(len(x)))

    x = np.array(x)
    x = x.reshape(x.shape[0], jump_data_length, num_columns, 1)

    pred = model.predict(x)

    pred = pd.DataFrame(pred, columns=['1 3/4 Salto vw B', '1 3/4 Salto vw C', '1/2 ein 1/2 aus C', '3/4 Salto rw A', '3/4 Salto vw A',
                    'Baby- Fliffis C', 'Barani A', 'Barani B', 'Barani C', 'Bauchsprung', 'Bücksprung', 'Cody C',
                    'Doppelsalto B', 'Doppelsalto C', 'Fliffis B', 'Fliffis C', 'Fliffis aus B', 'Fliffis aus C',
                    'Fliffis- Rudi B', 'Fliffis- Rudi C', 'Grätschwinkel', 'Halb ein Triffis C', 'Hocksprung', 'Rudi',
                    'Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C', 'Schraubensalto',
                    'Schraubensalto A', 'Schraubensalto C', 'Strecksprung', 'Triffis B', 'Triffis C', 'Voll- ein 1 3/4 Salto vw C',
                    'Voll- ein- Rudi- aus B', 'Voll- ein- halb- aus B', 'Voll- ein- voll- aus A', 'Voll- ein- voll- aus B',
                    'Voll- ein- voll- aus C', 'Von Bauch in Stand'])
    pred = np.array(pred.idxmax(axis=1))

    return pred


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

    if only_pp is None:
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


def run_multiple_times_DFF(num_columns, num_classes, runs, act_func, loss, optim, epochs, x_train, y_train, x_test, y_test):
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


def gen_shap_DFF(model, sample_train, sample_test, x_train, y_train, x_test, y_test):
    """
    generates the shap values for DFF

    Parameters
    ----------
    model : keras.model
        DFF model
    sample_train : int
        integer number. how often each jump should be sampled into the train data. We used 6
    sample_test : int
        integer number. how often each jump should be sampled into the test data. We used 3
    x_train : pandas.Dataframe
        return value of prepare_data_DFF
    y_train : pandas.Dataframe
        return value of prepare_data_DFF
    x_test : pandas.Dataframe
        return value of prepare_data_DFF
    y_test : pandas.Dataframe
        return value of prepare_data_DFF

    :return: shap_values
    """

    shap_x_train, shap_y_train = sample_x_test(x_train, y_train, sample_train)
    shap_x_test, shap_y_test = sample_x_test(x_test, y_test, sample_test)

    explainer = shap.KernelExplainer(model, shap_x_train)
    shap_values = explainer.shap_values(shap_x_test)

    with open("plots/CNN/shp_values.pkl", 'wb') as f:
        pickle.dump([shap_values, shap_x_train, shap_y_train, shap_x_test, shap_y_test], f)

    return shap_values


def predict_DFF(model, data, pp_list):
    """
    Predicts the classes of given x data

    Parameters
    ----------
    model : keras.model
        a DFF model
    data : pandas.Dataframe
        data to be predicted

    :return: prediction
    """

    first_djumps = set([col for col in data.columns if 'DJump' in col]) \
                   - set([col for col in data.columns if 'DJump_SIG_I_S' in col]) \
                   - set([col for col in data.columns if 'DJump_ABS_I_S' in col]) \
                   - set([col for col in data.columns if 'DJump_I_ABS_S' in col])
    if 1 not in pp_list:
        data = data.drop(first_djumps, axis=1)
    if 2 not in pp_list:
        data = data.drop([col for col in data.columns if 'DJump_SIG_I_S' in col], axis=1)
    if 3 not in pp_list:
        data = data.drop([col for col in data.columns if 'DJump_ABS_I_S' in col], axis=1)
    if 4 not in pp_list:
        data = data.drop([col for col in data.columns if 'DJump_I_ABS_S' in col], axis=1)

    data = data.drop(['SprungID'], axis=1)   # TODO: create an ID

    x = np.array(data)

    pred = model.predict(x)

    pred = pd.DataFrame(pred, columns=['1 3/4 Salto vw B', '1 3/4 Salto vw C', '1/2 ein 1/2 aus C', '3/4 Salto rw A',
                                       '3/4 Salto vw A',
                                       'Baby- Fliffis C', 'Barani A', 'Barani B', 'Barani C', 'Bauchsprung',
                                       'Bücksprung', 'Cody C',
                                       'Doppelsalto B', 'Doppelsalto C', 'Fliffis B', 'Fliffis C', 'Fliffis aus B',
                                       'Fliffis aus C',
                                       'Fliffis- Rudi B', 'Fliffis- Rudi C', 'Grätschwinkel', 'Halb ein Triffis C',
                                       'Hocksprung', 'Rudi',
                                       'Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C',
                                       'Schraubensalto',
                                       'Schraubensalto A', 'Schraubensalto C', 'Strecksprung', 'Triffis B', 'Triffis C',
                                       'Voll- ein 1 3/4 Salto vw C',
                                       'Voll- ein- Rudi- aus B', 'Voll- ein- halb- aus B', 'Voll- ein- voll- aus A',
                                       'Voll- ein- voll- aus B',
                                       'Voll- ein- voll- aus C', 'Von Bauch in Stand'])
    pred = np.array(pred.idxmax(axis=1))

    return pred
