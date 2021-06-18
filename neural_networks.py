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


def prepare_data():

    data_train = pd.read_csv("Sprungdaten_processed/with_preprocessed/percentage/20/percentage_mean_std_20_train.csv")
    data_test = pd.read_csv("Sprungdaten_processed/with_preprocessed/percentage/20/percentage_mean_std_20_test.csv")

    # data_train = pd.read_csv("Sprungdaten_processed/percentage/5/val/percentage_mean_5_80_10_10_train.csv")
    # data_test = pd.read_csv("Sprungdaten_processed/percentage/5/val/percentage_mean_5_80_10_10_test.csv")
    # data_val = pd.read_csv("Sprungdaten_processed/percentage/5/val/percentage_mean_5_80_10_10_val.csv")

    # DJump_SIG_I_S , DJump_ABS_I_S , DJump_I_ABS_S
    first_djumps = set([col for col in data_train.columns if 'DJump' in col]) - set([col for col in data_train.columns if 'DJump_SIG_I_S' in col])\
    - set([col for col in data_train.columns if 'DJump_ABS_I_S' in col]) - set([col for col in data_train.columns if 'DJump_I_ABS_S' in col])
    pp_list = [3]
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
    #x_val = []
    #y_val = []

    num_columns = len(data_train.columns) - 2

    jump_data_length = len(data_train[data_train['SprungID'] == data_train['SprungID'].unique()[0]])

    for id in data_train['SprungID'].unique():
        subframe = data_train[data_train['SprungID'] == id]
        y_train.append(subframe['Sprungtyp'].unique()[0])
        # 'Time', 'TimeInJump',
        subframe = subframe.drop(['SprungID', 'Sprungtyp'], axis=1)
        x_train.append(subframe)
        print("Preparing train data: " + str(len(x_train)))

    for id in data_test['SprungID'].unique():
        subframe = data_test[data_test['SprungID'] == id]
        y_test.append(subframe['Sprungtyp'].unique()[0])
        subframe = subframe.drop(['SprungID', 'Sprungtyp'], axis=1)
        x_test.append(subframe)
        print("Preparing test data: " + str(len(x_test)))

    """
    for id in data_val['SprungID'].unique():
        subframe = data_val[data_val['SprungID'] == id]
        y_val.append(subframe['Sprungtyp'].unique()[0])
        subframe = subframe.drop(['Time', 'TimeInJump', 'SprungID', 'Sprungtyp'], axis=1)
        x_val.append(subframe)
        print("Preparing validation data: " + str(len(x_val)))
    """

    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], jump_data_length, num_columns, 1)
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], jump_data_length, num_columns, 1)
    #x_val = np.array(x_val)
    #x_val = x_val.reshape(x_val.shape[0], jump_data_length, num_columns, 1)

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    #y_val = pd.get_dummies(y_val)

    #val = (x_val, y_val)

    return x_train, y_train, x_test, y_test, jump_data_length, num_columns


def prepare_data_oneliner():

    data_train = pd.read_csv("Sprungdaten_processed/with_preprocessed/percentage/25/vector_percentage_mean_std_25_train.csv")
    data_test = pd.read_csv("Sprungdaten_processed/with_preprocessed/percentage/25/vector_percentage_mean_std_25_test.csv")

    first_djumps = set([col for col in data_train.columns if 'DJump' in col]) - set([col for col in data_train.columns if 'DJump_SIG_I_S' in col]) \
    - set([col for col in data_train.columns if 'DJump_ABS_I_S' in col]) - set([col for col in data_train.columns if 'DJump_I_ABS_S' in col])
    pp_list = [3]
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

    num_columns = len(x_train.columns)

    y_train = data_train['Sprungtyp']
    y_test = data_test['Sprungtyp']

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    return x_train, y_train, x_test, y_test, num_columns


def build_model(jump_data_length, num_columns, c, kernel, pool, d, act_func, loss, optim):

    first_input = Input(shape=(jump_data_length, num_columns, 1), name="first_input")
    x = Conv2D(32, kernel_size=(kernel, kernel), padding="same", activation=act_func)(first_input)
    for i in range(c):
        x = Conv2D(32 * (i + 1), kernel_size=(kernel, kernel), padding="same", activation=act_func)(x)
    x = MaxPooling2D(pool_size=(pool, pool), padding="same")(x)
    x = Flatten()(x)
    for i in range(d):
        x = Dense(32 * d, activation=act_func)(x)
    x = Dense(45, activation='softmax', name="output")(x)

    model = Model(inputs=first_input, outputs=x)
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(), keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def build_model_grid(jump_data_length=20, num_columns=16, c=1, act_func='tanh', loss='categorical_crossentropy', optim='Nadam'):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation=act_func, input_shape=(jump_data_length, num_columns, 1)))
    for i in range(c):
        model.add(Conv2D(32 * (i + 1), kernel_size=(3, 3), padding="same", activation=act_func))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(64, activation=act_func))
    model.add(Dense(45, activation='softmax', name="output"))

    """first_input = Input(shape=(jump_data_length, num_columns, 1), name="first_input")
    x = Conv2D(32, kernel_size=(3, 3), padding="same", activation=act_func)(first_input)
    x = Conv2D(64, kernel_size=(3, 3), padding="same", activation=act_func)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    #x = Dropout(dropout_rate)
    x = Flatten()(x)
    x = Dense(64, activation=act_func)(x)
    x = Dense(45, activation='softmax', name="output")(x)

    model = Model(inputs=first_input, outputs=x)"""
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(), keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def grid_search_build(x_train, x_test, y_train, y_test, model, param_grid, cv=10, scoring_fit='neg_mean_squared_error'):

    gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, scoring=scoring_fit, verbose=2)
    fitted_model = gs.fit(x_train, y_train)

    pred = fitted_model.predict(x_test)

    return fitted_model, pred


def run_multiple_times(jump_data_length, num_columns, runs, conv, kernel, pool, dense, act_func, loss, optim, epochs, x_train, y_train, x_test, y_test):

    best_score = 0
    mean_score = 0

    for i in range(runs):
        #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=6)
        model = build_model(jump_data_length, num_columns, conv, kernel, pool, dense, act_func, loss, optim)
        #model = build_model_testing(jump_data_length, num_columns)
        model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1)
        score = model.evaluate(x_test, y_test, verbose=1)
        mean_score += score[1]
        if score[1] > best_score:
            best_score = score[1]
            best_scores = score
            best_model = model
    mean_score = mean_score / runs

    print("Best score: %f, Mean score: %f"%(best_score * 100, mean_score * 100))

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


def build_model_oneliner(num_columns, act_func, loss, optim):

    first_input = Input(shape=(num_columns,), name="first_input")
    x = Dense(128, activation=act_func)(first_input)
    x = Dense(256, activation=act_func)(x)
    x = Dense(512, activation=act_func)(x)
    x = Dense(512, activation=act_func)(x)
    x = Dense(512, activation=act_func)(x)
    x = Dense(256, activation=act_func)(x)
    x = Dense(128, activation=act_func)(x)
    x = Dense(45, activation='softmax', name="output")(x)

    model = Model(inputs=first_input, outputs=x)
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(), keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def run_multiple_times_oneliner(num_columns, runs, act_func, loss, optim, epochs, x_train, y_train, x_test, y_test):

    best_score = 0
    mean_score = 0

    for i in range(runs):
        #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=6)
        model = build_model_oneliner(num_columns, act_func, loss, optim)
        model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1)
        score = model.evaluate(x_test, y_test, verbose=1)
        mean_score += score[1]
        if score[1] > best_score:
            best_score = score[1]
            best_scores = score
            best_model = model
    mean_score = mean_score / runs

    print("Best score: %f, Mean score: %f"%(best_score * 100, mean_score * 100))

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


def sample_x_test(x_test, y_test):
    df = x_test.copy()
    df['Sprungtyp'] = y_test.idxmax(axis=1)
    counts = df['Sprungtyp'].value_counts()
    counts = counts.where(counts < 3, other=3)
    x = pd.DataFrame(columns=df.columns)

    for jump in df['Sprungtyp'].unique():
        subframe = df[df['Sprungtyp'] == jump]
        x = x.append(subframe.sample(counts[jump], random_state=1), ignore_index=True)

    x = x.sample(frac=1)        # shuffle
    y = x['Sprungtyp']
    x = x.drop(['Sprungtyp'], axis=1)
    for column in x.columns:
        x[column] = x[column].astype(float).round(3)

    return x, y


def main():

    #x_train, y_train, x_test, y_test, jump_data_length, num_columns = prepare_data()
    x_train, y_train, x_test, y_test, num_columns = prepare_data_oneliner()

    # model = grid_search_build(x_train, y_train, x_test, y_test, jump_data_length)
    # model = run_multiple_times(10, jump_data_length, 3, 3, 2, 2, 'tanh', 'kl_divergence', 'Nadam', x_train, y_train, x_test, y_test, 20)
    #model = run_multiple_times(jump_data_length, num_columns, runs=1, conv=1, kernel=3, pool=2, dense=2, act_func='tanh', loss='categorical_crossentropy', optim='Nadam', epochs=40, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    #model = run_multiple_times(jump_data_length, num_columns, runs=1, conv=3, kernel=3, pool=2, dense=2, act_func='tanh', loss='kl_divergence', optim='Nadam', epochs=40, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    model = run_multiple_times_oneliner(num_columns, runs=10, act_func='relu', loss='categorical_crossentropy', optim='Nadam', epochs=60, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    """
    param_grid = {'jump_data_length': [jump_data_length], 'num_columns': [num_columns], 'epochs': [40],
                  'batch_size': [32], 'optim': ['adam', 'Nadam'], 'c': [1, 2, 3],
                  'act_func': ['tanh', 'relu'], 'loss': ['categorical_crossentropy', 'kl_divergence']}
    model = KerasClassifier(build_fn=build_model_grid, verbose=0)
    model, pred = grid_search_build(x_train, x_test, y_train, y_test, model, param_grid, cv=2, scoring_fit='neg_log_loss')

    print(model.best_score_)
    print(model.best_params_)
    """
    """
    # randomized Grid Search for cnn
    param_grid = {'num_columns': [num_columns], 'epochs': [40],
                  'batch_size': [32], 'optim': ['adam', 'Nadam', 'SGD'],
                  'act_func': ['tanh', 'relu', 'sigmoid'], 'loss': ['categorical_crossentropy', 'kl_divergence']}
    model = KerasClassifier(build_fn=build_model_oneliner, verbose=0)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, verbose=1, n_iter=3, n_jobs=1)
    grid_result = grid.fit(x_test, y_test)
    cv_results_df = pd.DataFrame(grid_result.cv_results_)
    # cv_results_df.to_csv('gridsearch.csv')
    print(grid_result.best_params_)
    print(cv_results_df)  # via debugger
    """
    """
    model = run_multiple_times(jump_data_length, num_columns, runs=5, conv=3, kernel=3, pool=2, dense=2,
                               act_func='tanh', loss='kl_divergence', optim='Nadam', epochs=40,
                               x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    model.evaluate(x_test, y_test, verbose=1)
    """

    #model.evaluate(x_test, y_test, verbose=1)
    shap.initjs()
    #"""
    # DFF
    shap_x_test, y = sample_x_test(x_test, y_test)
    background = shap.sample(x_train, 400, random_state=1)
    explainer = shap.KernelExplainer(model, background)
    shap_values = explainer.shap_values(shap_x_test)

    cmap = ['#393b79','#5254a3','#6b6ecf','#9c9ede','#637939','#8ca252','#b5cf6b','#cedb9c','#8c6d31','#bd9e39','#e7ba52',
     '#e7cb94','#843c39','#ad494a','#d6616b','#e7969c','#7b4173','#a55194','#ce6dbd','#de9ed6','#3182bd','#6baed6',
     '#9ecae1','#c6dbef','#e6550d','#fd8d3c','#fdae6b','#fdd0a2','#31a354','#74c476','#a1d99b','#c7e9c0','#756bb1',
     '#9e9ac8','#bcbddc','#dadaeb','#636363','#969696','#969696','#d9d9d9','#f0027f','#f781bf','#f7b6d2','#fccde5',
     '#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']

    # hv.extension('matplotlib')
    # ListedColormap(process_cmap('winter'))
    shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(15, 17), color=ListedColormap(cmap), class_names=y.unique(), max_display=20)
    shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(15, 17), color=ListedColormap(cmap), class_names=y.unique(), max_display=68)
    shap.summary_plot(shap_values[0], shap_x_test, plot_size=(12, 12), title=y[0])
    #"""


    # CNN
    """
    
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(x_test[0: 5])
    shap.image_plot(shap_values, -x_test[0: 5]) #, labels=list(y_test.columns))
    
    # Shap for specific Class
    i = y_test.index[y_test['Salto C'] == 1]
    pd.DataFrame(model.predict(x_test[i]), columns=y_test.columns).idxmax(axis=1)
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(x_test[i])
    shap.image_plot(shap_values, -x_test[i])  # , labels=list(y_test.columns))
    
    # Confusion matrix to find mistakes in classification
    cm = sklearn.metrics.confusion_matrix(y_test.idxmax(axis=1), pd.DataFrame(model.predict(x_test), columns=y_test.columns).idxmax(axis=1))
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_test.columns)
    disp.plot(cmap='flag')
    disp.figure_.set_figwidth(30)
    disp.figure_.set_figheight(25)
    disp.figure_.autofmt_xdate()
    plt.show()
    #plt.savefig('CNN_confusion_matrix_flag.png')

    disp.plot(cmap='Blues')
    disp.figure_.set_figwidth(30)
    disp.figure_.set_figheight(25)
    disp.figure_.autofmt_xdate()
    plt.show()
    #plt.savefig('CNN_confusion_matrix.png')
    """


    return


if __name__ == '__main__':
    main()
