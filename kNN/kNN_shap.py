import pandas as pd
import shap
import numpy as np
from holoviews.plotting.util import process_cmap
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn import neighbors
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix
from random_classifier import metrics as rc_metrics
import time


def sample_x_test(x_test, y_test, num):
    df = x_test.copy()
    df['Sprungtyp'] = y_test
    counts = df['Sprungtyp'].value_counts()
    counts = counts.where(counts < num, other=num)
    x = pd.DataFrame(columns=df.columns)

    for jump in df['Sprungtyp'].unique():
        subframe = df[df['Sprungtyp'] == jump]
        x = x.append(subframe.sample(counts[jump], random_state=1), ignore_index=True)

    x = x.sample(frac=1)        # shuffle
    y = x['Sprungtyp']
    y = y.reset_index(drop=True)
    x = x.drop(['Sprungtyp'], axis=1)
    for column in x.columns:
        x[column] = x[column].astype(float).round(3)

    return x, y


if __name__ == '__main__':

    '''train_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')'''
    train_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')

    # get_features (X)
    '''start_column: str = 'DJump_SIG_I_x LapEnd'
    end_column: str = '90_std_Gyro_z_Fil' '''

    start_column: str = '0_mean_Acc_N_Fil'
    end_column: str = '90_std_Gyro_z_Fil'

    ''' p = train_data.drop([col for col in train_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    t = test_data.drop([col for col in test_data.columns if 'DJump_SIG_I_S' in col], axis=1)  '''

    X_train = train_data.loc[:, start_column:end_column]
    y_train = (train_data['Sprungtyp'])

    X_test = test_data.loc[:, start_column:end_column]
    y_test = (test_data['Sprungtyp'])

    # params
    n_neighbors = 3
    weights = 'distance'
    dist_metric = 'manhattan'

    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=dist_metric, p=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    if accuracy_score(y_test, y_pred) > 0:
        print(f"Accuracy: ", metrics.accuracy_score(y_test, y_pred).__round__(4))
        mean_prec, mean_rec, mean_f, mean_youden = rc_metrics(y_test, y_pred)
        print(f"Accuracy youden score: {str(mean_youden.round(4))}")
        print(f"Accuracy f1 score: {str(mean_f.round(4))}")
        print("--------------------------------------------------------------")

    # print(y_train.to_frame())

    '''explainer = shap.KernelExplainer(clf.decision_function, X_test.sample(n=50), link='identity')
    shap_values = explainer.shap_values(X_test.sample(n=10))
    shap.summary_plot(shap_values[0], X_test.sample(n=10))'''

    # colormap
    cmap = ['#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31',
            '#bd9e39', '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194', '#ce6dbd',
            '#de9ed6', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354',
            '#74c476', '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363', '#969696', '#969696',
            '#d9d9d9', '#f0027f', '#f781bf', '#f7b6d2', '#fccde5', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
            '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f']

    cmap_cm = process_cmap('summer')
    cmap_cm.insert(0, '#ffffff')
    cmap_cm.insert(-1, '#000000')
    cmap_cm = ListedColormap(cmap_cm)

    shap_x_test, shap_y_test = sample_x_test(X_test, y_test, 3)
    shap_x_train, shap_y_train = sample_x_test(X_train, y_train, 6)

    ''' f = lambda x: clf.predict_proba(x)[:, 1]
    explainer = shap.KernelExplainer(f, shap_x_train)
    shap_values = explainer.shap_values(shap_x_test)'''

    f = lambda x: clf.predict_proba(x)[:, 1]
    # med = shap_x_test.median().values.reshape((1, shap_x_test.shape[1]))
    explainer = shap.KernelExplainer(f, shap.kmeans(shap_x_train, 10))
    shap_values = explainer.shap_values(shap_x_test)

    print(shap_values)

    shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(20, 17), color=ListedColormap(cmap),
                      class_names=shap_y_test.unique(), max_display=20)
    shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(15, 17), color=ListedColormap(cmap),
                      class_names=shap_y_test.unique(), max_display=68)
   
    saltoA = np.where(shap_y_test.unique() == 'Salto A')[0][0]
    shap.summary_plot(shap_values[saltoA], shap_x_test, plot_size=(12, 12), title='Salto A')
    saltoB = np.where(shap_y_test.unique() == 'Salto B')[0][0]
    shap.summary_plot(shap_values[saltoB], shap_x_test, plot_size=(12, 12), title='Salto B')
    saltoC = np.where(shap_y_test.unique() == 'Salto C')[0][0]
    shap.summary_plot(shap_values[saltoC], shap_x_test, plot_size=(12, 12), title='Salto C')

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(20, 20))
    plot_confusion_matrix(clf, shap_x_test, shap_y_test, xticks_rotation='vertical', display_labels=set(y_test), cmap=plt.cm.Blues, normalize=None, ax = ax)
    plt.show()