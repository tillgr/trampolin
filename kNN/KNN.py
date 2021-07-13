import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

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

def random_search_all_parameters(data_train, data_test, pp_list):
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

                print(
                    f"Accuracy score: {losses} , {penalty} , {maxi}:  {str(accuracy_score(y_test, y_pred).__round__(4))}")
                mean_prec, mean_rec, mean_f, mean_youden = metrics(y_test, y_pred)
                print(f"Accuracy youden score: {str(mean_youden.round(4))}")
                print(f"Accuracy f1 score: {str(mean_f.round(4))}")


def test_all_datasets_with_all_parameters(pp_list):
    for i in [1, 2, 5, 10, 20, 25]:
        for calc_type in ['', 'mean_', 'mean_std_']:
            print(f"Folder: {i}")
            print(f"> Type: {calc_type}")
            data_train = pd.read_csv("../Sprungdaten_processed/with_preprocessed/percentage/" + str(
                i) + "/vector_percentage_" + calc_type + str(i) + "_train.csv")
            data_test = pd.read_csv("../Sprungdaten_processed/with_preprocessed/percentage/" + str(
                i) + "/vector_percentage_" + calc_type + str(i) + "_test.csv")

            random_search_all_parameters(data_train, data_test, pp_list)


def get_best_data_set_with_preprocessed():
    train_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/25/vector_percentage_25_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/with_preprocessed/percentage/25/vector_percentage_25_test.csv')
    train_data = train_data.drop([col for col in train_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    test_data = test_data.drop([col for col in test_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    return train_data, test_data


def get_best_data_set_without_preprocessed():
    train_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/20/vector_percentage_mean_20_train.csv')
    test_data = pd.read_csv(
        '../Sprungdaten_processed/without_preprocessed/percentage/20/vector_percentage_mean_20_test.csv')
    return train_data, test_data


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


def gbc_classifier(X_train, y_train, X_test, estimators: str, depth: str):
    clf = GradientBoostingClassifier(n_estimators=estimators, max_depth=depth)

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    return clf, y_pred

def shap_plots(data_train, data_test, pp_list, estimators, depth, aj=None):
    X_train, y_train, X_test, y_test = prepare_data(data_train, data_test, pp_list)
    clf, y_pred = gbc_classifier(X_train, y_train, X_test, estimators, depth)

    for dataType in ['with', 'without']:
        for aj in ['', 'AJ']:
            print("---")
            print(dataType)
            print(aj)

            #todo
            if dataType == 'with':
                train_data = pd.read_csv(
                    '../Sprungdaten_processed/with_preprocessed/percentage/25/vector_percentage_25_train.csv')

                test_data = pd.read_csv(
                    '../Sprungdaten_processed/with_preprocessed/percentage/25/vector_percentage_25_test.csv')
                if aj == 'AJ':
                    test_data = pd.read_csv(
                        '../Sprungdaten_processed/with_preprocessed/percentage/25/vector_AJ_percentage_25.csv')

            if dataType == 'without':
                train_data = pd.read_csv(
                    '../Sprungdaten_processed/without_preprocessed/percentage/20/vector_percentage_mean_20_train.csv')

                test_data = pd.read_csv(
                    '../Sprungdaten_processed/without_preprocessed/percentage/20/vector_percentage_mean_20_test.csv')
                if aj == 'AJ':
                    test_data = pd.read_csv(
                        '../Sprungdaten_processed/without_preprocessed/percentage/20/vector_AJ_percentage_mean_20.csv')

            # get_features (X)
            if dataType == 'with':
                start_column: str = '0_Acc_N_Fil'
                end_column: str = '75_Gyro_z_Fil'

            if dataType == 'without':
                start_column: str = '0_Acc_N_Fil'
                end_column: str = '80_Gyro_z_Fil'


            X_train = train_data.loc[:, start_column:end_column]
            y_train = (train_data['Sprungtyp'])

            X_test = test_data.loc[:, start_column:end_column]
            y_test = (test_data['Sprungtyp'])

            # params
            if dataType == 'with':
                estimators = 200
                depth = 7

            if dataType == 'without':
                estimators = 90
                depth = 3

            clf = GradientBoostingClassifier(n_estimators=estimators, max_depth=depth)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            cmap_cm = process_cmap('summer')
            cmap_cm.insert(0, '#ffffff')
            cmap_cm.insert(-1, '#000000')
            cmap_cm = ListedColormap(cmap_cm)

            cmap_cm_AJ = ['#ffffff', '#048166']
            cmap_cm_AJ = ListedColormap(cmap_cm_AJ)

            shap_x_test, shap_y_test = sample_x_test(X_test, y_test, 3)
            shap_x_train, shap_y_train = sample_x_test(X_train, y_train, 6)

            explainer = shap.KernelExplainer(clf.predict_proba, shap_x_train)
            shap_values = explainer.shap_values(shap_x_test)


            bar_plots(shap_values, shap_x_test, shap_y_test,
                      bar='percentual',
                      folder='../plots/GBC/' + dataType + '_preprocessed/',
                      name=aj + '_' + dataType + '_preprocessed')
            bar_plots(shap_values, shap_x_test, shap_y_test,
                      bar='percentual',
                      jumps=['Salto A', 'Salto B', 'Salto C', 'Salto rw A', 'Salto rw B', 'Salto rw C',
                             'Schraubensalto',
                             'Schraubensalto A', 'Schraubensalto C', 'Doppelsalto B', 'Doppelsalto C'],
                      folder='../plots/GBC/' + dataType + '_preprocessed/',
                      name=aj + '_Saltos')
            bar_plots(shap_values, shap_x_test, shap_y_test,
                      folder='../plots/GBC/' + dataType + '_preprocessed/',
                      name=aj + '_' + dataType + '_preprocessed')

            # confusion matrix
            if dataType == 'with':
                if aj == 'AJ':
                    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
                    disp.plot(cmap=cmap_cm_AJ)
                    disp.figure_.set_figwidth(35)
                    disp.figure_.set_figheight(25)
                    disp.figure_.autofmt_xdate()
                    plt.tick_params(axis='x', labelsize=10, labelrotation=45, grid_linewidth=5)
                    plt.title("GBC/with_preprocessed/vector_AJ_percentage_25_train")
                    plt.tight_layout()
                    plt.savefig('../plots/GBC/with_preprocessed/cf_AJ')
                    plt.show()

                if aj == '':
                    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
                    disp.plot(cmap=cmap_cm)
                    disp.figure_.set_figwidth(35)
                    disp.figure_.set_figheight(25)
                    disp.figure_.autofmt_xdate()
                    plt.tick_params(axis='x', labelsize=10, labelrotation=45, grid_linewidth=5)
                    plt.title("GBC/with_preprocessed/vector_percentage_25_train")
                    plt.tight_layout()
                    plt.savefig('../plots/GBC/with_preprocessed/cf_noAJ')
                    plt.show()

            if dataType == 'without':
                if aj == 'AJ':
                    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
                    disp.plot(cmap=cmap_cm_AJ)
                    disp.figure_.set_figwidth(35)
                    disp.figure_.set_figheight(25)
                    disp.figure_.autofmt_xdate()
                    plt.tick_params(axis='x', labelsize=10, labelrotation=45, grid_linewidth=5)
                    plt.title("GBC/without_preprocessed/vector_AJ_percentage_mean_20_train")
                    plt.tight_layout()
                    plt.savefig('../plots/GBC/without_preprocessed/cf_AJ')
                    plt.show()

                if aj == '':
                    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
                    disp.plot(cmap=cmap_cm)
                    disp.figure_.set_figwidth(35)
                    disp.figure_.set_figheight(25)
                    disp.figure_.autofmt_xdate()
                    plt.tick_params(axis='x', labelsize=10, labelrotation=45, grid_linewidth=5)
                    plt.title("GBC/without_preprocessed/vector_percentage_mean_20_train")
                    plt.tight_layout()
                    plt.savefig('../plots/GBC/without_preprocessed/cf_noAJ')
                    plt.show()


def jump_core_detection(data_train, data_test, pp_list, jump_length=0):
    scores = {}
    percentage = int(100 / jump_length)
    full_list = [l for l in range(0, 100, percentage)]

    variants = []
    for i in range(jump_length - 1):
        variants.append([l for l in range(i + 1)])
    for i in range(1, jump_length):
        variants.append(list(range(i, jump_length)))
    for i in range(int((jump_length - 1) / 2)):
        both_sides = list(set(list(range(jump_length))) - set(list(range(0, jump_length))[i + 1:-1 - i]))
        both_sides.sort()
        variants.append(both_sides)
    print(variants)

    for variant in variants:
        indexes = []
        for i in range(int(len(data_train) / jump_length)):
            for to_delete in variant:
                indexes.append(i * jump_length + to_delete)
        data_train_copy = data_train.drop(indexes)
        print(data_train_copy)
        indexes = []
        for i in range(int(len(data_test) / jump_length)):
            for to_delete in variant:
                indexes.append(i * jump_length + to_delete)
        data_test_copy = data_test.drop(indexes)
        X_train, y_train, X_test, y_test = prepare_data(data_train_copy, data_test_copy, pp_list)
        clf = GradientBoostingClassifier(n_estimators=90, max_depth=3)
        y_pred = clf.predict(X_test)
        print(f"Accuracy score:  {str(accuracy_score(y_test, y_pred).__round__(4))}")
        score = accuracy_score(y_test, y_pred).__round__(4)
        print(score)

        variant_output = [v * percentage for v in variant]
        variant_output = list(set(full_list) - set(variant_output))
        variant_output.sort()

        scores[str(variant_output[0]) + ' - ' + str(variant_output[-1])] = score

    print(f"scores:  {scores}")

    min_y_value = 70
    plt.figure(figsize=(13, 13))
    plt.suptitle('GBC without pp: percentage_mean_20')
    plt.xlabel('Data')
    plt.ylabel('Accuracy')
    plt.axis([0, full_list[-1], min_y_value, 100])
    plt.xticks(range(0, 100 + percentage, percentage))
    plt.yticks(range(min_y_value, 105, 5))
    plt.grid(True, axis='x')
    # cmap = process_cmap('brg', len(scores))

    for i in range(len(scores)):
        entry = list(scores.items())[i]
        start, end = entry[0].split('-')
        acc = entry[1] * 100
        print(acc)
        print(min_y_value)
        if int(acc) >= min_y_value:
            if start.replace(' ', '') == '0':
                plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#0000ff', alpha=0.7)
            elif end.replace(' ', '') == str(full_list[-1]):
                plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#ff0000', alpha=0.7)
            else:
                plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#00ff00', alpha=0.7)

    plt.show()


if __name__ == '__main__':
    train_data = pd.read_csv('../Sprungdaten_processed/without_preprocessed/percentage/20/vector_percentage_mean_20_train.csv')
    test_data = pd.read_csv('../Sprungdaten_processed/without_preprocessed/percentage/20/vector_percentage_mean_20_test.csv')
    jump_core_detection(train_data, test_data, [1, 2, 3, 4], 5)