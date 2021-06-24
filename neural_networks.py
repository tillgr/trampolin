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


def prepare_data(data_train, data_test, pp_list):

    first_djumps = set([col for col in data_train.columns if 'DJump' in col]) - set([col for col in data_train.columns if 'DJump_SIG_I_S' in col])\
    - set([col for col in data_train.columns if 'DJump_ABS_I_S' in col]) - set([col for col in data_train.columns if 'DJump_I_ABS_S' in col])
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
        x_train.append(subframe)
        print("Preparing train data: " + str(len(x_train)))

    for id in data_test['SprungID'].unique():
        subframe = data_test[data_test['SprungID'] == id]
        y_test.append(subframe['Sprungtyp'].unique()[0])
        subframe = subframe.drop(['SprungID', 'Sprungtyp'], axis=1)
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


def prepare_data_oneliner(data_train, data_test, pp_list):

    first_djumps = set([col for col in data_train.columns if 'DJump' in col]) - set([col for col in data_train.columns if 'DJump_SIG_I_S' in col]) \
    - set([col for col in data_train.columns if 'DJump_ABS_I_S' in col]) - set([col for col in data_train.columns if 'DJump_I_ABS_S' in col])
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

    num_classes = len(y_train.columns)

    return x_train, y_train, x_test, y_test, num_columns, num_classes


def build_model(jump_data_length, num_columns, num_classes, c, kernel, pool, d, act_func, loss, optim):

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
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(), keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def build_model_grid(jump_data_length=20, num_columns=16, num_classes=43, c=1, act_func='tanh', loss='categorical_crossentropy', optim='Nadam'):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation=act_func, input_shape=(jump_data_length, num_columns, 1)))
    for i in range(c):
        model.add(Conv2D(32 * (i + 1), kernel_size=(3, 3), padding="same", activation=act_func))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(64, activation=act_func))
    model.add(Dense(num_classes, activation='softmax', name="output"))

    model.compile(loss=loss, optimizer=optim, metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(), keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def grid_search_build(x_train, x_test, y_train, y_test, model, param_grid, cv=10, scoring_fit='neg_mean_squared_error'):

    gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, scoring=scoring_fit, verbose=2)
    fitted_model = gs.fit(x_train, y_train)

    pred = fitted_model.predict(x_test)

    return fitted_model, pred


def run_multiple_times(jump_data_length, num_columns, num_classes, runs, conv, kernel, pool, dense, act_func, loss, optim, epochs, x_train, y_train, x_test, y_test):

    best_score = 0
    mean_score = 0

    for i in range(runs):
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
        model = build_model(jump_data_length, num_columns, num_classes, conv, kernel, pool, dense, act_func, loss, optim)
        #model = build_model_testing(jump_data_length, num_columns)
        model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1, callbacks=[callback])
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


def build_model_oneliner(num_columns, num_classes, act_func, loss, optim):

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
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(), keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def run_multiple_times_oneliner(num_columns, num_classes, runs, act_func, loss, optim, epochs, x_train, y_train, x_test, y_test):

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


def sample_x_test(x_test, y_test, num, cnn=False):

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

        x = x.sample(frac=1, random_state=1)        # shuffle
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
        df['Sprungtyp'] = y_test.idxmax(axis=1)     # if not one-hot-encoded:   df['Sprungtyp'] = y_test
        counts = df['Sprungtyp'].value_counts()
        counts = counts.where(counts < num, other=num)
        x = pd.DataFrame(columns=df.columns)

        for jump in df['Sprungtyp'].unique():
            subframe = df[df['Sprungtyp'] == jump]
            x = x.append(subframe.sample(counts[jump], random_state=1), ignore_index=True)

        x = x.sample(frac=1, random_state=1)        # shuffle
        y = x['Sprungtyp']
        y = y.reset_index(drop=True)
        x = x.drop(['Sprungtyp'], axis=1)
        for column in x.columns:
            x[column] = x[column].astype(float).round(3)

    return x, y


def get_index(jump_list, y_test):
    index_list = []
    for jump in jump_list:
        index_list.extend(y_test.where(y_test == jump).dropna().index.tolist())
    return index_list


def create_colormap(shap_y_test, shap_values=None):
    """
    takes values and creates unique colormap, so that each jump, have always the sam color
    :param shap_values: output from explainer.shap_values(shap_x_test)
    :param shap_y_test: sampled y_test or single string jump
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
        'Bücksprung': '#8c6d31',
        'Grätschwinkel': '#bd9e39',
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


def main():
    neural_network = 'dff'  # 'dff'  'cnn'
    run_modus = ''     # 'multi' 'grid'
    run = 50                # for multi runs or how often random grid search runs
    data_train = pd.read_csv("Sprungdaten_processed/with_preprocessed/percentage/25/vector_percentage_mean_std_25_train.csv")
    data_test = pd.read_csv("Sprungdaten_processed/with_preprocessed/percentage/25/vector_percentage_mean_std_25_test.csv")
    pp_list = [3]

    if neural_network == 'cnn':
        x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes = prepare_data(data_train, data_test, pp_list)
        if run_modus == 'multi':
            model = run_multiple_times(jump_data_length, num_columns, num_classes, runs=run, conv=3, kernel=3, pool=2, dense=2,
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

    if neural_network == 'dff':
        x_train, y_train, x_test, y_test, num_columns, num_classes = prepare_data_oneliner(data_train, data_test, pp_list)
        if run_modus == 'multi':
            model = run_multiple_times_oneliner(num_columns, num_classes, runs=run, act_func='relu', loss='categorical_crossentropy',
                                                optim='Nadam', epochs=100, x_train=x_train, y_train=y_train,
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

    #model.save("models/DFF_without_mean_std_20")
    model = keras.models.load_model("models/DFF_with_mean_std_25")
    model.summary()
    model.evaluate(x_test, y_test, verbose=1)
    shap.initjs()
    # colormap
    cmap = ['#393b79','#5254a3','#6b6ecf','#9c9ede','#637939','#8ca252','#b5cf6b','#cedb9c','#8c6d31','#bd9e39','#e7ba52',
     '#e7cb94','#843c39','#ad494a','#d6616b','#e7969c','#7b4173','#a55194','#ce6dbd','#de9ed6','#3182bd','#6baed6',
     '#9ecae1','#c6dbef','#e6550d','#fd8d3c','#fdae6b','#fdd0a2','#31a354','#74c476','#a1d99b','#c7e9c0','#756bb1',
     '#9e9ac8','#bcbddc','#dadaeb','#636363','#969696','#969696','#d9d9d9','#f0027f','#f781bf','#f7b6d2','#fccde5',
     '#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']

    cmap_cm = process_cmap('summer')
    cmap_cm.insert(0, '#ffffff')
    cmap_cm.insert(-1, '#000000')
    cmap_cm = ListedColormap(cmap_cm)


    cmap_cm_AJ = ['#ffffff', '#048166']
    cmap_cm_AJ = ListedColormap(cmap_cm_AJ)

    # ListedColormap(process_cmap('winter'))

    #"""
    # DFF
    salto_i = y_test.index[y_test['Salto C'] == 1]

    shap_x_test, shap_y_test = sample_x_test(x_test, y_test, 3)
    shap_x_train, shap_y_train = sample_x_test(x_train, y_train, 6)
    #background = shap.sample(shap_x_train, 400, random_state=1)
    explainer = shap.KernelExplainer(model, shap_x_train)
    shap_values = explainer.shap_values(shap_x_test)

    bar_cm = create_colormap(shap_y_test, shap_values)

    shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(25, 20), color=bar_cm, class_names=shap_y_test.unique(), max_display=20)
    shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(25, 20), color=bar_cm, class_names=shap_y_test.unique(), max_display=68)

    saltoA = np.where(shap_y_test.unique() == 'Salto A')[0][0]
    shap.summary_plot(shap_values[saltoA], shap_x_test, plot_size=(25, 15), title='Salto A')

    saltoB = np.where(shap_y_test.unique() == 'Salto B')[0][0]
    shap.summary_plot(shap_values[saltoB], shap_x_test, plot_size=(25, 15), title='Salto B')
    saltoC = np.where(shap_y_test.unique() == 'Salto C')[0][0]
    shap.summary_plot(shap_values[saltoC], shap_x_test, plot_size=(25, 15), title='Salto C')

    # creating vor each jump a bar chart
    for jump in shap_y_test.unique():
        color_string = create_colormap(jump)
        shap.summary_plot(shap_values[np.where(shap_y_test.unique() == jump)[0][0]].__abs__(), shap_x_test, plot_type='bar', color=color_string, plot_size=(20,20), show=False)
        plt.savefig('plots/DFF/with_preprocessed/jump_analysis/' + jump.replace('/', '-') + '.png')
        plt.clf()

    #"""


    # CNN
    """
    shap_x_test, shap_y_test = sample_x_test(x_test, y_test, 3, cnn=True)
    shap_x_train, shap_y_train = sample_x_test(x_train, y_train, 6, cnn=True)

    parts = {1: ['1 3/4 Salto vw B', '1 3/4 Salto vw C', '1/2 ein 1/2 aus C', '3/4 Salto rw A', '3/4 Salto vw A', 'Baby- Fliffis C'],
             2: ['Barani A', 'Barani B', 'Barani C', 'Cody C', 'Rudi'],
             3: ['Bauchsprung', 'Bücksprung', 'Grätschwinkel', 'Hocksprung', 'Von Bauch in Stand', 'Strecksprung'],
             4: ['Fliffis B', 'Fliffis C', 'Fliffis aus B', 'Fliffis aus C', 'Fliffis- Rudi B', 'Fliffis- Rudi C'],
             5: ['Halb ein Triffis C', 'Triffis B', 'Triffis C'],
             6: ['Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C'],
             7: ['Schraubensalto', 'Schraubensalto A', 'Schraubensalto C',  'Doppelsalto B', 'Doppelsalto C'],
             8: ['Voll- ein 1 3/4 Salto vw C', 'Voll- ein- Rudi- aus B', 'Voll- ein- halb- aus B', 'Voll- ein- voll- aus A', 'Voll- ein- voll- aus B', 'Voll- ein- voll- aus C']}

    for j in [1, 2, 3, 4, 5, 6, 7, 8]:
        index_list = get_index(parts[j], shap_y_test)
        to_explain = shap_x_test[index_list]
        explainer = shap.DeepExplainer(model, shap_x_train)
        shap_values, indexes = explainer.shap_values(to_explain, ranked_outputs=4, check_additivity=False)
        d = dict(enumerate(np.array(y_test.columns).flatten(), 0))
        index_names = np.vectorize(lambda i: d[i])(indexes)

        shap.image_plot(shap_values, to_explain, index_names, show=False)
        plt.savefig('plots/CNN/with_preprocessed/CNN_with_mean_std_20_part' + str(j) + '.png')
    """
    """
    # Shap for specific Class
    i = y_test.index[y_test['Salto C'] == 1]
    pd.DataFrame(model.predict(x_test[i]), columns=y_test.columns).idxmax(axis=1)
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(x_test[i])
    shap.image_plot(shap_values, -x_test[i])  # , labels=list(y_test.columns))
    """

    #"""
    # Confusion matrix to find mistakes in classification
    cm = sklearn.metrics.confusion_matrix(y_test.idxmax(axis=1), pd.DataFrame(model.predict(x_test), columns=y_test.columns).idxmax(axis=1))
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_test.columns)
    disp.plot(cmap=cmap_cm)
    disp.figure_.set_figwidth(35)
    disp.figure_.set_figheight(25)
    disp.figure_.autofmt_xdate()
    plt.show()
    #plt.savefig('CNN_confusion_matrix_flag.png')
    #"""


    return


if __name__ == '__main__':
    main()
