import numpy as np
import pandas as pd
import shap
from holoviews.plotting.util import process_cmap
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, ConfusionMatrixDisplay
from neural_networks import create_colormap, bar_plots
from random_classifier import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sgd import sample_x_test

if __name__ == '__main__':

    train_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')
    '''train_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')'''

    # get_features (X)
    start_column: str = 'DJump_SIG_I_x LapEnd'
    end_column: str = 'DJump_ABS_I_S4_z LapEnd'

    '''start_column: str = list(train_data.columns)[2]
    end_column: str = list(train_data.columns)[-1]'''

    '''start_aj: str = list(test_data.columns)[0]
    end_aj: str = list(test_data.columns)[-3]'''

    # p = train_data.drop([col for col in train_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    # t = test_data.drop([col for col in test_data.columns if 'DJump_SIG_I_S' in col], axis=1)

    # features
    X_train = train_data.loc[:, start_column:end_column]
    # targets
    y_train = train_data['Sprungtyp']

    X_test = test_data.loc[:, start_column:end_column]
    y_test = test_data['Sprungtyp']

    clf = SGDClassifier(loss='log', penalty='l1', alpha=0.0001,
                        l1_ratio=0.15, fit_intercept=True, max_iter=10000,
                        tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                        n_jobs=None, random_state=10, learning_rate='optimal',
                        eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
                        n_iter_no_change=5, class_weight=None,
                        warm_start=False, average=False).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if accuracy_score(y_test, y_pred) > 0:
        print(f"Accuracy score:  {str(accuracy_score(y_test, y_pred).__round__(4))}")
        mean_prec, mean_rec, mean_f, mean_youden = metrics(y_test, y_pred)
        print(f"Accuracy youden score: {str(mean_youden.round(4))}")
        print(f"Accuracy f1 score: {str(mean_f.round(4))}")

    cmap_cm = process_cmap('summer')
    cmap_cm.insert(0, '#ffffff')
    cmap_cm.insert(-1, '#000000')
    cmap_cm = ListedColormap(cmap_cm)

    # cmap_cm_AJ = ['#ffffff', '#048166']
    # cmap_cm_AJ = ListedColormap(cmap_cm_AJ)

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
    # fig, ax = plt.subplots(figsize=(20, 20))
    # plot_confusion_matrix(clf, shap_x_test, shap_y_test, xticks_rotation='vertical', display_labels=set(y_test), cmap=plt.cm.Blues, normalize=None, ax = ax)
    # plt.show()
