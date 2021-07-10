import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from random_classifier import metrics
from holoviews.plotting.util import process_cmap
from matplotlib.colors import ListedColormap
from neural_networks import bar_plots
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def prepare_data(data_train, data_test, pp_list):
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

    X_train = data_train.drop('Sprungtyp', axis=1)
    X_train = X_train.drop(['SprungID'], axis=1)
    X_test = data_test.drop('Sprungtyp', axis=1)
    X_test = X_test.drop(['SprungID'], axis=1)

    y_train = data_train['Sprungtyp']
    y_test = data_test['Sprungtyp']

    return X_train, y_train, X_test, y_test


def all_parameters_classifier(data_train, data_test, pp_list, accuracy):
    X_train, y_train, X_test, y_test = prepare_data(data_train, data_test, pp_list)
    for losses in ['log', 'modified_huber', 'squared_hinge', 'perceptron']:
        for penalty in ['l2', 'l1', 'elasticnet']:
            for maxi in [10000]:
                clf = SGDClassifier(loss=losses, penalty=penalty, alpha=0.0001,
                                    l1_ratio=0.15, fit_intercept=True, max_iter=maxi,
                                    tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                                    n_jobs=None, random_state=10, learning_rate='optimal',
                                    eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                                    n_iter_no_change=5, class_weight=None,
                                    warm_start=False, average=False).fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                if accuracy_score(y_test, y_pred) > accuracy:
                    print(
                        f"Accuracy score: {losses} , {penalty} , {maxi}:  {str(accuracy_score(y_test, y_pred).__round__(4))}")
                    mean_prec, mean_rec, mean_f, mean_youden = metrics(y_test, y_pred)
                    print(f"Accuracy youden score: {str(mean_youden.round(4))}")
                    print(f"Accuracy f1 score: {str(mean_f.round(4))}")


def all_datasets_with_all_parameters_classifier(pp_list, accuracy):
    for i in [1, 2, 5, 10, 20, 25]:
        for calc_type in ['', 'mean_', 'mean_std_']:
            print(f"Folder: {i}")
            print(f"> Type: {calc_type}")
            data_train = pd.read_csv("../Sprungdaten_processed/with_preprocessed/percentage/" + str(
                i) + "/vector_percentage_" + calc_type + str(i) + "_train.csv")
            data_test = pd.read_csv("../Sprungdaten_processed/with_preprocessed/percentage/" + str(
                i) + "/vector_percentage_" + calc_type + str(i) + "_test.csv")

            all_parameters_classifier(data_train, data_test, pp_list, accuracy)


def get_best_data_set_with_preprocessed():
    train_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')
    train_data = train_data.drop([col for col in train_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    test_data = test_data.drop([col for col in test_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    return train_data, test_data


def get_best_data_set_without_preprocessed():
    train_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')
    return train_data, test_data


def get_targets(data):
    return data['Sprungtyp']


def sample_x_test(x_test, y_test, num):
    df = x_test.copy()
    df['Sprungtyp'] = y_test
    counts = df['Sprungtyp'].value_counts()
    counts = counts.where(counts < num, other=num)
    x = pd.DataFrame(columns=df.columns)

    for jump in df['Sprungtyp'].unique():
        subframe = df[df['Sprungtyp'] == jump]
        x = x.append(subframe.sample(counts[jump], random_state=1), ignore_index=True)

    x = x.sample(frac=1)  # shuffle
    y = x['Sprungtyp']
    y = y.reset_index(drop=True)
    x = x.drop(['Sprungtyp'], axis=1)
    for column in x.columns:
        x[column] = x[column].astype(float).round(3)

    return x, y


def sgd_classifier(X_train, y_train, X_test, loss: str, penalty: str, max_iter: int):
    clf = SGDClassifier(loss=loss, penalty=penalty, alpha=0.0001,
                        l1_ratio=0.15, fit_intercept=True, max_iter=max_iter,
                        tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                        n_jobs=None, random_state=10, learning_rate='optimal',
                        eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                        n_iter_no_change=5, class_weight=None,
                        warm_start=False, average=False).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred


def shap_plots(data_train, data_test, pp_list, loss, penalty, max_iter, aj=None):
    X_train, y_train, X_test, y_test = prepare_data(data_train, data_test, pp_list)
    clf, y_pred = sgd_classifier(X_train, y_train, X_test, loss, penalty, max_iter)

    if aj is None:
        cmap_cm = process_cmap('summer')
        cmap_cm.insert(0, '#ffffff')
        cmap_cm.insert(-1, '#000000')
        cmap_cm = ListedColormap(cmap_cm)

        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(cmap=cmap_cm)
        disp.figure_.set_figwidth(35)
        disp.figure_.set_figheight(25)
        disp.figure_.autofmt_xdate()
        plt.tick_params(axis='x', labelsize=10, labelrotation=45, grid_linewidth=5)
        plt.title("SGD_with_mean_std_10_ConfusionMatrix")
        plt.tight_layout()
        plt.savefig('../plots/SGD/with_preprocessed/only_preprocessed')
        plt.show()

    else:
        cmap_cm_AJ = ['#ffffff', '#048166']
        cmap_cm_AJ = ListedColormap(cmap_cm_AJ)

        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(cmap=cmap_cm_AJ)
        disp.figure_.set_figwidth(35)
        disp.figure_.set_figheight(25)
        disp.figure_.autofmt_xdate()
        plt.tick_params(axis='x', labelsize=10, labelrotation=45, grid_linewidth=5)
        plt.title("SGD_with_mean_std_10_ConfusionMatrix_AJ")
        plt.tight_layout()
        plt.savefig('../plots/SGD/with_preprocessed/only_preprocessed/AJ')
        plt.show()

    shap_x_test, shap_y_test = sample_x_test(X_test, y_test, 3)
    shap_x_train, shap_y_train = sample_x_test(X_train, y_train, 6)

    explainer = shap.KernelExplainer(clf.decision_function, shap_x_train)
    shap_values = explainer.shap_values(shap_x_test)

    bar_plots(shap_values, shap_x_test, shap_y_test, bar='percentual', size=(50, 30),
              folder='../plots/SGD/with_preprocessed/only_preprocessed', name='only_with_preprocessed')

    bar_plots(shap_values, shap_x_test, shap_y_test, bar='percentual', size=(50, 30),
              jumps=['Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C', 'Schraubensalto',
                     'Schraubensalto A', 'Schraubensalto C', 'Doppelsalto B', 'Doppelsalto C'],
              folder='../plots/SGD/with_preprocessed/only_preprocessed', name='Saltos_only_with_preprocessed')

    bar_plots(shap_values, shap_x_test, shap_y_test, size=(30, 45), bar='summary',
              folder='../plots/SGD/with_preprocessed/only_preprocessed',
              name='only_with_preprocessed')

    saltoA = np.where(shap_y_test.unique() == 'Salto A')[0][0]
    shap.summary_plot(shap_values[saltoA], shap_x_test, plot_size=(30, 12), title='Salto A')
    saltoB = np.where(shap_y_test.unique() == 'Salto B')[0][0]
    shap.summary_plot(shap_values[saltoB], shap_x_test, plot_size=(30, 12), title='Salto B')
    saltoC = np.where(shap_y_test.unique() == 'Salto C')[0][0]
    shap.summary_plot(shap_values[saltoC], shap_x_test, plot_size=(30, 12), title='Salto C')


