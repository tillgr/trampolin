import numpy as np
import pandas as pd
import shap
from holoviews.plotting.util import process_cmap
from matplotlib.colors import ListedColormap
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from random_classifier import metrics
from sklearn import preprocessing


def sample_x_test(x_test, y_test):
    df = x_test.copy()
    df['Sprungtyp'] = x_test.idxmax(axis=1)
    counts = df['Sprungtyp'].value_counts()
    counts = counts.where(counts < 3, other=3)
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

    train_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')

    # get_features (X)
    start_column: str = 'DJump_SIG_I_x LapEnd'
    end_column: str = '90_std_Gyro_z_Fil'

    p = train_data.drop([col for col in train_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    t = test_data.drop([col for col in test_data.columns if 'DJump_SIG_I_S' in col], axis=1)

    X_train = p.loc[:, start_column:end_column]
    y_train = (p['Sprungtyp'])

    X_test = t.loc[:, start_column:end_column]
    y_test = (t['Sprungtyp'])

    clf = SGDClassifier(loss='log', penalty='l1', alpha=0.0001,
                        l1_ratio=0.15, fit_intercept=True, max_iter=10000,
                        tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                        n_jobs=None, random_state=10, learning_rate='optimal',
                        eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                        n_iter_no_change=5, class_weight=None,
                        warm_start=False, average=False).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if accuracy_score(y_test, y_pred) > 0.9:
        print(f"Accuracy score:  {str(accuracy_score(y_test, y_pred).__round__(4))}")
        mean_prec, mean_rec, mean_f, mean_youden = metrics(y_test, y_pred)
        print(f"Accuracy youden score: {str(mean_youden.round(4))}")
        print(f"Accuracy f1 score: {str(mean_f.round(4))}")

    #print(y_train.to_frame())

    '''explainer = shap.KernelExplainer(clf.decision_function, X_test.sample(n=50), link='identity')
    shap_values = explainer.shap_values(X_test.sample(n=10))
    shap.summary_plot(shap_values[0], X_test.sample(n=10))'''


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
    shap_x_test, y = sample_x_test(X_test, y_test)
    background = shap.sample(X_train, 400, random_state=1)
    explainer = shap.KernelExplainer(clf.decision_function, background)
    shap_values = explainer.shap_values(shap_x_test)

    shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(15, 17), color=ListedColormap(cmap),
                      class_names=y.unique(), max_display=20)
    shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(15, 17), color=ListedColormap(cmap),
                      class_names=y.unique(), max_display=68)
    shap.summary_plot(shap_values[0], shap_x_test, plot_size=(12, 12), title=y[0])
