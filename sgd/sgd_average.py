import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from random_classifier import metrics

if __name__ == '__main__':

    train_data = pd.read_csv('../Sprungdaten_processed/with_preprocessed/std_data/std_data_train.csv')
    test_data = pd.read_csv('../Sprungdaten_processed/with_preprocessed/std_data/std_data_test.csv')

    # get_features (X)
    start_column: str = 'std_Acc_N_Fil'
    end_column: str = 'std_Gyro_z_Fil'

    p = train_data.drop([col for col in train_data.columns if 'DJump_SIG_I_S' in col], axis=1)
    t = test_data.drop([col for col in test_data.columns if 'DJump_SIG_I_S' in col], axis=1)

    X_train = p.loc[:, start_column:end_column].to_numpy()
    y_train = np.array((train_data['Sprungtyp']))

    X_test = t.loc[:, start_column:end_column].to_numpy()
    y_test = np.array((test_data['Sprungtyp']))

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

    #explainer = shap.KernelExplainer(clf.decision_function, X_test.sample(n=50), link='identity')
    #shap_values = explainer.shap_values(X_test.sample(n=10))
    #shap.summary_plot(shap_values[0], X_test.sample(n=10))
