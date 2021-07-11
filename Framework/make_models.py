import pandas as pd
import Framework.neural_networks as nn
import keras


def train_CNN_standard(preprocessed='with_preprocessed'):

    if preprocessed == 'without_preprocessed':
        data_train = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/5/percentage_mean_5_train.csv')
        data_test = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/5/percentage_mean_5_test.csv')
        params = [10, 3, 3, 2, 2, 'tanh', 'kl_divergence', 'Nadam', 40]

    elif preprocessed == 'with_preprocessed':
        data_train = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/20/percentage_mean_std_20_train.csv')
        data_test = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/20/percentage_mean_std_20_train.csv')
        params = [10, 3, 3, 2, 2, 'tanh', 'categorical_crossentropy', 'Nadam', 40]

    else:
        raise AttributeError('preprocessed not correctly defined!')

    pp_list = [3]

    runs, conv, kernel, pool, dense, act_func, loss, optim, epochs = [params[i] for i in range(len(params))]

    x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes = nn.prepare_data_CNN(data_train, data_test, pp_list)
    model = nn.run_multiple_times_CNN(jump_data_length, num_columns, num_classes,
                                      runs, conv, kernel, pool, dense, act_func, loss, optim, epochs,
                                      x_train, y_train, x_test, y_test)

    keras.models.save_model(model, "models/CNN_with_mean_std_20")


def train_DFF_standard(preprocessed='without_preprocessed'):

    if preprocessed == 'without_preprocessed':
        data_train = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/20/vector_percentage_mean_std_20_train.csv')
        data_test = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/20/vector_percentage_mean_std_20_test.csv')
        params = [12, 'relu', 'categorical_crossentropy', 'Nadam', 100]

    elif preprocessed == 'with_preprocessed':
        data_train = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/25/vector_percentage_mean_std_25_train.csv')
        data_test = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/25/vector_percentage_mean_std_25_train.csv')
        params = [12, 'relu', 'categorical_crossentropy', 'Nadam', 100]

    else:
        raise AttributeError('preprocessed not correctly defined!')

    pp_list = [3]

    runs, conv, act_func, loss, optim, epochs = [params[i] for i in range(len(params))]

    x_train, y_train, x_test, y_test, num_columns, num_classes = nn.prepare_data_DFF(data_train, data_test, pp_list)
    model = nn.run_multiple_times_DFF(num_columns, num_classes, runs, act_func, loss, optim, epochs,
                                      x_train, y_train, x_test, y_test)

    keras.models.save_model(model, "models/DFF_without_mean_std_20")


if __name__ == '__main__':

    print("You can now choose which models to train. Both are used in the predictions")
    print("Please make sure the needed data is in Sprungdaten_preprocessed (ran make_data.py)")
    print("Keep in mind this will take some time")
    print("Which model do you want to train? [dff / cnn / both]")
    x = input()

    if x == 'dff' or x == 'DFF' or x == 'Dff':
        train_DFF_standard()

    elif x == 'cnn' or x == 'CNN' or x == 'Cnn':
        train_CNN_standard()

    elif x == 'both' or x == 'BOTH' or x == 'Both':
        train_DFF_standard()
        train_CNN_standard()

    else:
        raise AttributeError("You did not specify which models to train")
