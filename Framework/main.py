import Framework.process_data as process_data
import Framework.neural_networks as nn
import Framework.plots as plots
import pandas as pd
import keras


def create_all_data():

    process_data.make_all_data()
    process_data.make_data_only_jumps()
    for p in ['with_preprocessed', 'without_preprocessed']:
        for m in [None, 'mean', 'mean_std']:
            process_data.make_percentage_data(p, ['0.25', '0.20', '0.10', '0.05', '0.02', '0.01'], m)
    for p in ['with_preprocessed', 'without_preprocessed']:
        process_data.make_avg_std_data(p)
    for p in ['with_preprocessed', 'without_preprocessed']:
        for n in ['percentage_mean_std', 'percentage_mean', 'percentage']:
            process_data.make_AJ_jumps(p, n, ['25', '20', '10', '5', '2', '1'])


def create_used_data():

    process_data.make_all_data()
    process_data.make_data_only_jumps()

    for p in ['with_preprocessed', 'without_preprocessed']:
        for m in [None, 'mean', 'mean_std']:
            process_data.make_percentage_data(p, ['0.25', '0.20', '0.10', '0.05'], m)


def train_CNN(data_train, data_test, pp_list, params, only_pp=False):

    runs, conv, kernel, pool, dense, act_func, loss, optim, epochs = [params[i] for i in range(len(params))]

    x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes = nn.prepare_data_CNN(data_train, data_test, pp_list, only_pp)
    model = nn.run_multiple_times_CNN(jump_data_length, num_columns, num_classes,
                              runs, conv, kernel, pool, dense, act_func, loss, optim, epochs,
                              x_train, y_train, x_test, y_test)

    return model


def train_CNN_standard(preprocessed='without_preprocessed'):

    if preprocessed == 'without_preprocessed':
        data_train = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/5/percentage_mean_5_train.csv')
        data_test = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/5/percentage_mean_5_test.csv')
        params = [5, 3, 3, 2, 2, 'tanh', 'kl_divergence', 'Nadam', 40]

    elif preprocessed == 'with_preprocessed':
        data_train = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/20/percentage_mean_std_20_train.csv')
        data_test = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/20/percentage_mean_std_20_train.csv')
        params = [5, 3, 3, 2, 2, 'tanh', 'categorical_crossentropy', 'Nadam', 40]

    else:
        raise AttributeError('preprocessed not correctly defined!')

    pp_list = [3]

    runs, conv, kernel, pool, dense, act_func, loss, optim, epochs = [params[i] for i in range(len(params))]

    x_train, y_train, x_test, y_test, jump_data_length, num_columns, num_classes = nn.prepare_data_CNN(data_train, data_test, pp_list)
    model = nn.run_multiple_times_CNN(jump_data_length, num_columns, num_classes,
                                      runs, conv, kernel, pool, dense, act_func, loss, optim, epochs,
                                      x_train, y_train, x_test, y_test)

    return model


def train_DFF(data_train, data_test, pp_list, params, only_pp=False):

    runs, conv, act_func, loss, optim, epochs = [params[i] for i in range(len(params))]

    x_train, y_train, x_test, y_test, num_columns, num_classes = nn.prepare_data_DFF(data_train, data_test, pp_list, only_pp)
    model = nn.run_multiple_times_DFF(num_columns, num_classes, runs, act_func, loss, optim, epochs,
                                      x_train, y_train, x_test, y_test)

    return model


def train_DFF_standard(preprocessed='without_preprocessed'):

    if preprocessed == 'without_preprocessed':
        data_train = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/20/vector_percentage_mean_std_20_train.csv')
        data_test = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/20/vector_percentage_mean_std_20_test.csv')
        params = [8, 'relu', 'categorical_crossentropy', 'Nadam', 100]

    elif preprocessed == 'with_preprocessed':
        data_train = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/25/vector_percentage_mean_std_25_train.csv')
        data_test = pd.read_csv('Sprungdaten_processed/' + preprocessed + '/percentage/25/vector_percentage_mean_std_25_train.csv')
        params = [8, 'relu', 'categorical_crossentropy', 'Nadam', 100]

    else:
        raise AttributeError('preprocessed not correctly defined!')

    pp_list = [3]

    runs, conv, act_func, loss, optim, epochs = [params[i] for i in range(len(params))]

    x_train, y_train, x_test, y_test, num_columns, num_classes = nn.prepare_data_DFF(data_train, data_test, pp_list)
    model = nn.run_multiple_times_DFF(num_columns, num_classes, runs, act_func, loss, optim, epochs,
                                      x_train, y_train, x_test, y_test)

    return model


def main():

    print("Please choose what you want to do:")
    print("Create all data: 1")
    print("Create data used for training the best models: 2")
    print("Train a standard CNN: 3")
    print("Train a standard DFF: 4")
    print("Make a prediction with CNN: 5")
    print("Make a prediction with DFF: 6")
    print("Create plots of the models: 7")

    i = input()

    if i == '1':        # Create all data
        create_all_data()

    elif i == '2':      # Create used data
        create_used_data()

    elif i == '3':      # Train standard CNN
        train_CNN_standard()

    elif i == '4':      # Train standard DFF
        train_DFF_standard()

    elif i == '5':      # Make prediction with CNN
        print("Please specify which dataset to use:")

        model = keras.models.load_model('../models/CNN_without_mean_5')
        pred = nn.predict_CNN(model, "Sprungdaten_processed/without_preprocessed/percentage/5/percentage_mean_5_test.csv")
        print(pred)

    elif i == '6':      # Make prediction with DFF
        print("Please specify which dataset to use:")

        model = keras.models.load_model('../models/DFF_without_mean_std_20')
        pred = nn.predict_DFF(model, "Sprungdaten_processed/without_preprocessed/percentage/20/vector_percentage_mean_std_20_test.csv")
        print(pred)

    elif i == '7':      # Create plots
        print("Please specify which plot to create:")
        print("Confusion Matrix: 1")
        print("Jump Core Analysis: 2")
        print("Beeswarm: 3")
        print("Image plot: 4")
        print("Bar plots: 5")

        i = input()

        if i == '1':  # Confusion Matrix


        elif i == '2':  # Jump Core Analysis


        elif i == '3':  # Beeswarm


        elif i == '4':  # Image plot


        elif i == '4':  # Bar plots


        else:
            print("Selected option not available.")

    else:
        print("Selected option not available.")

    """
    data_train = pd.read_csv('Sprungdaten_processed/without_preprocessed/percentage/5/percentage_mean_5_train.csv')
    data_test = pd.read_csv('Sprungdaten_processed/without_preprocessed/percentage/5/percentage_mean_5_test.csv')
    params = [3, 3, 3, 2, 2, 'tanh', 'kl_divergence', 'Nadam', 40]
    pp_list = [3]

    #model = train_CNN(data_train, data_test, pp_list, params)
    """


if __name__ == '__main__':
    main()
