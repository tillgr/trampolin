import numpy as np
import pandas as pd
import shap
from holoviews.plotting.util import process_cmap
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, ConfusionMatrixDisplay
from neural_networks import create_colormap
from random_classifier import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


def bar_plots(shap_values, shap_x_test, shap_y_test, bar=None, jumps=None, folder=None, name=None):
    """

    :param shap_values:
    :param shap_x_test:
    :param shap_y_test:
    :param bar: different bar plots, can be 'summary', 'single' or 'percentual'
    :param jumps: for 'single' and 'percentual' you can imput a list with the jump names, for that bar plots should be created
    :param folder: path, where the plots should be saved, whens its empty, than plots will be shown and not be saved
    :param name: for saving the plot you can choose an other name
    :return:
    """
    if name is None:
        name = ''
    else:
        name = '-' + name
    if jumps is None:
        jumps = []
    if bar is None or bar == 'summary':
        if folder is None:
            shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(30, 40),
                              color=ListedColormap('#616161'), class_names=shap_y_test.unique(),
                              max_display=len(shap_x_test.columns))
        else:
            shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(30, 40),
                              color=ListedColormap('#616161'), class_names=shap_y_test.unique(),
                              max_display=len(shap_x_test.columns), show=False)
            plt.savefig(folder + 'summary_plot' + name + '.png')
        plt.clf()

    elif bar == 'single':
        if len(jumps) == 0:
            jumps = shap_y_test.unique()
        for jump in jumps:
            color_string = create_colormap(jump)
            shap.summary_plot(shap_values[np.where(shap_y_test.unique() == jump)[0][0]].__abs__(), shap_x_test,
                              plot_type='bar', color=color_string, plot_size=(20, 20), show=False)
            if folder is None:
                plt.show()
            else:
                plt.savefig(folder + jump.replace('/', '-') + '.png')
            plt.clf()
    elif bar == 'percentual':
        feature_names = shap_x_test.columns
        jump_dict = {}
        if len(jumps) == 0:
            jumps = shap_y_test.unique().tolist()
            jumps.sort()
        for jump in jumps:
            values = np.abs(shap_values[np.where(shap_y_test.unique() == jump)[0][0]].__abs__()).mean(0)
            sum_values = np.sum(values)
            jump_dict[jump] = [(v / sum_values) * 100 for v in values]

        # creates the plot
        labels = list(jump_dict.keys())
        data = np.array(list(jump_dict.values()))
        data_cum = data.cumsum(axis=1)
        # rainbow or turbo cmap
        category_colors = plt.get_cmap('rainbow')(
            np.linspace(0.15, 0.85, len([item for item in feature_names if item.startswith('0')])))
        # how often category_colors have to repeat itself
        number_sub_cm = len([item for item in feature_names if not item.startswith('DJump')]) / \
                        len([item for item in feature_names if item.startswith('0')])
        category_colors = np.tile(category_colors, (int(number_sub_cm), 1))

        count_djumps = len([item for item in feature_names if item.startswith('DJump')])
        # if Djumps in features, than other colormap must be appended
        if count_djumps > 0:

            dj_12_colors = np.array([
                [0.19216, 0.50980, 0.74118, 1.00000],
                [0.19216, 0.63922, 0.32941, 1.00000],
                [0.90196, 0.33333, 0.05098, 1.00000],
                [0.41961, 0.68235, 0.83922, 1.00000],
                [0.45490, 0.76863, 0.46275, 1.00000],
                [0.99216, 0.55294, 0.23529, 1.00000],
                [0.61961, 0.79216, 0.88235, 1.00000],
                [0.63137, 0.85098, 0.60784, 1.00000],
                [0.99216, 0.68235, 0.41961, 1.00000],
                [0.77647, 0.85882, 0.93725, 1.00000],
                [0.78039, 0.91373, 0.75294, 1.00000],
                [0.99216, 0.81569, 0.63529, 1.00000]])
            dj_9_colors = np.array([
                [0.48235, 0.25490, 0.45098, 1.00000],
                [0.64706, 0.31765, 0.58039, 1.00000],
                [0.80784, 0.42745, 0.74118, 1.00000],
                [0.45882, 0.41961, 0.69412, 1.00000],
                [0.61961, 0.60392, 0.78431, 1.00000],
                [0.73725, 0.74118, 0.86275, 1.00000],
                [0.88235, 0.07843, 0.50588, 1.00000],
                [0.96863, 0.50588, 0.74902, 1.00000],
                [0.96863, 0.73333, 0.82353, 1.00000]])

            # if nine djump pack not included
            if (count_djumps % 12) == 0:
                # if Djump features first, colormap must be appended on the first position
                if feature_names[0].startswith('DJump'):
                    category_colors = np.concatenate((
                        np.tile(dj_12_colors, (int(count_djumps / 12), 1)),
                        category_colors))
                else:
                    category_colors = np.concatenate((
                        category_colors,
                        np.tile(dj_12_colors, (int(count_djumps / 12), 1))))
            # if djump contains nine pack
            else:
                c = int(count_djumps - 9 / 12)
                sub_cm = np.concatenate((dj_9_colors, np.tile(dj_12_colors, (c, 1))))
                if feature_names[0].startswith('DJump'):
                    category_colors = np.concatenate((sub_cm, category_colors))
                else:
                    category_colors = np.concatenate((category_colors, sub_cm))

        fig, ax = plt.subplots(figsize=(30, 18))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(feature_names.tolist(), category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.6,
                            label=colname, color=color)

            r, g, b, _ = color
            # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # ax.bar_label(rects, label_type='center', color=text_color)
            ax.legend(ncol=1, bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0.)
        plt.suptitle('Percentual impact bar plot of the individual features')

        if folder is None:
            plt.show()
        else:
            plt.savefig(folder + 'percentual_plot' + name + '.png')
        plt.clf()
    return


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


if __name__ == '__main__':

    '''train_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/10/vector_AJ_percentage_mean_std_10.csv')'''
    train_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/10/vector_percentage_mean_std_10_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/10/vector_percentage_mean_std_10_test.csv')

    # get_features (X)
    '''start_column: str = 'DJump_SIG_I_x LapEnd'
    end_column: str = '90_std_Gyro_z_Fil' '''
    start_column: str = list(train_data.columns)[2]
    end_column: str = list(train_data.columns)[-1]

    # p = train_data.drop([col for col in train_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    # t = test_data.drop([col for col in test_data.columns if 'DJump_SIG_I_S' in col], axis=1)

    X_train = train_data.loc[:, start_column:end_column]
    y_train = (train_data['Sprungtyp']).to_numpy()

    X_test = test_data.loc[:, start_column:end_column]
    y_test = (test_data['Sprungtyp']).to_numpy()

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

    bar_plots(shap_values, shap_x_test, shap_y_test, bar='percentual',
              folder='../plots/SGD/without_preprocessed/', name='without_preprocessed')

    bar_plots(shap_values, shap_x_test, shap_y_test, bar='percentual',
              jumps=['Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C', 'Schraubensalto',
                     'Schraubensalto A', 'Schraubensalto C', 'Doppelsalto B', 'Doppelsalto C'],
              folder='../plots/SGD/without_preprocessed/', name='Saltos_without_preprocessed')
    bar_plots(shap_values, shap_x_test, shap_y_test, folder='../plots/SGD/without_preprocessed/',
              name='without_preprocessed')

    '''shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(20, 17), color=ListedColormap(cmap),
                      class_names=shap_y_test.unique(), max_display=20)
        shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=(15, 17), color=ListedColormap(cmap),
                      class_names=shap_y_test.unique(), max_display=68)'''

    '''saltoA = np.where(shap_y_test.unique() == 'Salto A')[0][0]
    shap.summary_plot(shap_values[saltoA], shap_x_test, plot_size=(12, 12), title='Salto A')
    saltoB = np.where(shap_y_test.unique() == 'Salto B')[0][0]
    shap.summary_plot(shap_values[saltoB], shap_x_test, plot_size=(12, 12), title='Salto B')
    saltoC = np.where(shap_y_test.unique() == 'Salto C')[0][0]
    shap.summary_plot(shap_values[saltoC], shap_x_test, plot_size=(12, 12), title='Salto C')'''

    '''cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap=cmap_cm_AJ)
    disp.figure_.set_figwidth(35)
    disp.figure_.set_figheight(25)
    disp.figure_.autofmt_xdate()
    plt.tick_params(axis='x', labelsize=10, labelrotation=45, grid_linewidth=5)
    plt.title("SGD_without_mean_std_10_ConfusionMatrix_AJ")
    plt.tight_layout()
    plt.savefig('../plots/SGD/without_preprocessed/AJ')
    plt.show()'''
    # fig, ax = plt.subplots(figsize=(20, 20))
    # plot_confusion_matrix(clf, shap_x_test, shap_y_test, xticks_rotation='vertical', display_labels=set(y_test), cmap=plt.cm.Blues, normalize=None, ax = ax)
    # plt.show()
