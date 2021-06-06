import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Dropout, SpatialDropout2D, AveragePooling2D
from keras import backend as k
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
import shap
import sklearn


def prepare_data():

    # data_train = pd.read_csv("Sprungdaten_processed/same_length/same_length_padding_0_train.csv")
    # data_test = pd.read_csv("Sprungdaten_processed/same_length/same_length_padding_0_test.csv")
    data_train = pd.read_csv("Sprungdaten_processed/percentage/2/percentage_mean_2_train.csv")
    data_test = pd.read_csv("Sprungdaten_processed/percentage/2/percentage_mean_2_test.csv")

    # data_train = pd.read_csv("Sprungdaten_processed/percentage/5/val/percentage_mean_5_80_10_10_train.csv")
    # data_test = pd.read_csv("Sprungdaten_processed/percentage/5/val/percentage_mean_5_80_10_10_test.csv")
    # data_val = pd.read_csv("Sprungdaten_processed/percentage/5/val/percentage_mean_5_80_10_10_val.csv")

    data_train = data_train.drop(['DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                  'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'], axis=1)
    data_test = data_test.drop(['DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                                  'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'], axis=1)

    data_train = data_train.drop(['ACC_N_ROT_filtered'], axis=1)
    data_test = data_test.drop(['ACC_N_ROT_filtered'], axis=1)

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

    data_train = pd.read_csv("Sprungdaten_processed/percentage/25/vector_percentage_25_train.csv")
    data_test = pd.read_csv("Sprungdaten_processed/percentage/25/vector_percentage_25_test.csv")

    x_train = data_train.drop('Sprungtyp', axis=1)
    x_train = x_train.drop(['SprungID', 'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                            'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'], axis=1)
    x_test = data_test.drop('Sprungtyp', axis=1)
    x_test = x_test.drop(['SprungID', 'DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
                          'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd'], axis=1)
    x_train = x_train.drop([col for col in x_train.columns if 'ACC_N_ROT_filtered' in col], axis=1)
    x_test = x_test.drop([col for col in x_test.columns if 'ACC_N_ROT_filtered' in col], axis=1)

    num_columns = len(x_train.columns)

    y_train = data_train['Sprungtyp']
    y_test = data_test['Sprungtyp']

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    return x_train, y_train, x_test, y_test, num_columns


def build_model_testing(jump_data_length, num_columns):

    first_input = Input(shape=(jump_data_length, num_columns, 1), name="first_input")
    x = Conv2D(221, kernel_size=(5, 5), padding="same", activation='tanh')(first_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(237, kernel_size=(3, 3), padding="same", activation='tanh')(x)
    x = Conv2D(207, kernel_size=(4, 4), padding="same", activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1438, activation='tanh')(x)
    x = Dense(1557, activation='tanh')(x)
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
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(), keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

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


def run_multiple_times(jump_data_length, num_columns, runs, conv, kernel, pool, dense, act_func, loss, optim, epochs, x_train, y_train, x_test, y_test):

    best_score = 0
    mean_score = 0

    for i in range(runs):
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        model = build_model(jump_data_length, num_columns, conv, kernel, pool, dense, act_func, loss, optim)
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


def build_model_oneliner(num_columns, act_func, loss, optim):

    first_input = Input(shape=(num_columns,), name="first_input")
    x = Dense(128, activation=act_func)(first_input)
    x = Dense(256, activation=act_func)(x)
    x = Dense(512, activation=act_func)(x)
    x = Dense(512, activation=act_func)(x)
    x = Dense(512, activation=act_func)(x)
    x = Dense(256, activation=act_func)(x)
    x = Dense(128, activation=act_func)(x)
    x = Dense(41, activation='softmax', name="output")(x)

    model = Model(inputs=first_input, outputs=x)
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy', keras.metrics.TruePositives(), keras.metrics.TrueNegatives(), keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    return model


def run_multiple_times_oneliner(num_columns, runs, act_func, loss, optim, epochs, x_train, y_train, x_test, y_test):

    best_score = 0
    mean_score = 0

    for i in range(runs):
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        model = build_model_oneliner(num_columns, act_func, loss, optim)
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


def main():

    x_train, y_train, x_test, y_test, jump_data_length, num_columns = prepare_data()
    #x_train, y_train, x_test, y_test, num_columns = prepare_data_oneliner()

    # model = grid_search_build(x_train, y_train, x_test, y_test, jump_data_length)
    # model = build_model_testing(jump_data_length, x_train, y_train)
    # model = run_multiple_times(10, jump_data_length, 3, 3, 2, 2, 'tanh', 'kl_divergence', 'Nadam', x_train, y_train, x_test, y_test, 20)
    model = run_multiple_times(jump_data_length, num_columns, runs=10, conv=1, kernel=3, pool=2, dense=2, act_func='tanh', loss='categorical_crossentropy', optim='Nadam', epochs=40, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    #model = run_multiple_times(jump_data_length, num_columns, runs=10, conv=3, kernel=3, pool=2, dense=2, act_func='tanh', loss='kl_divergence', optim='Nadam', epochs=30, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    #model = run_multiple_times_oneliner(num_columns, runs=1, act_func='tanh', loss='kl_divergence', optim='adam', epochs=60, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    model.evaluate(x_test, y_test, verbose=1)

    """
    # DFF
    shap.initjs()
    background = shap.sample(x_train, 100)
    explainer = shap.KernelExplainer(model, background)
    shap_values = explainer.shap_values(x_test, nsamples=100)

    shap.summary_plot(shap_values, x_test, plot_type='bar')
    shap.summary_plot(shap_values[0], x_test)
    shap.plots.force(explainer.expected_value[0], shap_values[0])
    shap.force_plot(explainer.expected_value[0], shap_values[0], x_test)
    """

    """
    # CNN
    shap.initjs()
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(x_test[1: 3])
    shap.image_plot(shap_values, -x_test[1: 3])
    """

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


    return


if __name__ == '__main__':
    main()
