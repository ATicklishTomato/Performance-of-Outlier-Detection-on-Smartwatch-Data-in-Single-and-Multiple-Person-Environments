import itertools
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from preprocessing import train_without_outliers


def train_test_validate_split(df, mix_data, within_subject):
    if within_subject:
        numpy_data = df.to_numpy()
        window_lengths = [np.linalg.norm(x) for x in numpy_data]
        outlier_indexes = np.argpartition(window_lengths, (-1 * int(0.1 * len(numpy_data))))[
                          (-1 * int(0.1 * len(numpy_data))):]
        outliers = df.iloc[outlier_indexes]
        data_without_outliers = pd.concat([df, outliers]).drop_duplicates(keep=False)
        data_without_outliers = pd.concat([data_without_outliers, mix_data], ignore_index=True)
    else:
        outliers = mix_data
        data_without_outliers = df

    # train test split
    X_train, X_test, y_train, y_test = train_without_outliers(data_without_outliers, outliers, int(len(df) * 0.2))

    # validation test split
    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def gmm_aic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the AIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.aic(X)


def find_best_params(X_validate, n_limit=15):
    param_grid = {
        "n_components": range(1, n_limit),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_aic_score
    )
    grid_search.fit(X_validate)

    return grid_search.best_params_['covariance_type'], grid_search.best_params_['n_components']


def roc_curving(gmm, X_validate, y_validate, print_curve=False):
    # Fit on GMM and determine scores
    scores = gmm.score_samples(X_validate.to_numpy())

    # Get ROC and AUC
    auc = roc_auc_score(y_validate.to_numpy(), np.array(scores))
    fpr, tpr, thresholds_sk = roc_curve(y_validate.to_numpy(), np.array(scores))
    if print_curve:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    optimal_score_cutoff = \
        sorted(list(zip(np.abs(np.subtract(tpr, fpr)), thresholds_sk)), key=lambda i: i[0], reverse=True)[0][1]
    return optimal_score_cutoff, auc


def prep_and_test(gmm, X_test, y_test, threshold=25):
    # Train and determine outliers
    scores = gmm.score_samples(X_test.to_numpy())

    predictions = pd.Series(scores).apply(lambda x: 0 if x < threshold else 1)
    TP = [1 for x, y in zip(predictions, y_test.tolist()) if x == 0 and y == 0]
    FP = [1 for x, y in zip(predictions, y_test.tolist()) if x == 0 and y == 1]
    TN = [1 for x, y in zip(predictions, y_test.tolist()) if x == 1 and y == 1]
    FN = [1 for x, y in zip(predictions, y_test.tolist()) if x == 1 and y == 0]
    return len(TP), len(FP), len(TN), len(FN)


def prep_train_test_data(dfs, mix_data, within_group):
    # Prep DataFrames for the GaussianMixture function and determine outliers
    if within_group:
        numpy_frames = [df.to_numpy() for df in dfs]
        all_outliers = []
        all_normal_data = []
        for df, numpy_data in zip(dfs, numpy_frames):
            window_lengths = [np.linalg.norm(x) for x in numpy_data]
            outlier_indexes = np.argpartition(window_lengths, (-1 * int(0.1 * len(numpy_data))))[
                              (-1 * int(0.1 * len(numpy_data))):]
            outliers = df.iloc[outlier_indexes]
            all_outliers.append(outliers)
            data_without_outliers = pd.concat([df, outliers]).drop_duplicates(keep=False)
            all_normal_data.append(data_without_outliers)
        all_normal_data.append(mix_data)
    else:
        all_outliers = [pd.concat(mix_data, ignore_index=True)]
        all_normal_data = [pd.concat(dfs, ignore_index=True)]

    # Create train batches from x number of normal windows of every subject, and a test set for testing
    x_train_data = []
    x_test_data = []
    x_validate_data = []
    y_validate_data = []
    y_train_data = []
    y_test_data = []
    for data_without_outliers, outliers in zip(all_normal_data, all_outliers):
        X_train, X_test, y_train, y_test = train_without_outliers(data_without_outliers, outliers,
                                                                  int(len(outliers) * 2))

        # validation test split
        X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        x_train_data.append(X_train.values.tolist())
        x_test_data.append(X_test.values.tolist())
        x_validate_data.append(X_validate.values.tolist())
        y_validate_data.append(y_validate.tolist())
        y_train_data.append(y_train)
        y_test_data.append(y_test.tolist())

    X_train = list(itertools.chain.from_iterable(x_train_data))
    X_test = list(itertools.chain.from_iterable(x_test_data))
    X_validate = list(itertools.chain.from_iterable(x_validate_data))
    y_validate = list(itertools.chain.from_iterable(y_validate_data))
    y_train = list(itertools.chain.from_iterable(y_train_data))
    y_test = list(itertools.chain.from_iterable(y_test_data))

    return np.array(X_train), np.array(X_validate), np.array(X_test), y_train, y_validate, y_test


def roc_curve_multiple(gmm, X_validate, y_validate, print_curve=False):
    scores = gmm.score_samples(X_validate)

    # Get ROC and AUC
    auc = roc_auc_score(y_validate, np.array(scores))
    fpr, tpr, thresholds_sk = roc_curve(y_validate, np.array(scores))
    if print_curve:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    optimal_score_cutoff = \
        sorted(list(zip(np.abs(np.subtract(tpr, fpr)), thresholds_sk)), key=lambda i: i[0], reverse=True)[0][1]
    return optimal_score_cutoff, auc


def train_and_test_multiple(gmm, X_test, y_test, threshold=33):
    # Train and determine outliers
    scores = gmm.score_samples(X_test)

    predictions = pd.Series(scores).apply(lambda x: 0 if x < threshold else 1)
    TP = [1 for x, y in zip(predictions, y_test) if x == 0 and y == 0]
    FP = [1 for x, y in zip(predictions, y_test) if x == 0 and y == 1]
    TN = [1 for x, y in zip(predictions, y_test) if x == 1 and y == 1]
    FN = [1 for x, y in zip(predictions, y_test) if x == 1 and y == 0]
    return len(TP), len(FP), len(TN), len(FN)


def run_single_gmm(df, mix_data, within_subject):
    X_train, X_validate, X_test, y_train, y_validate, y_test = train_test_validate_split(df, mix_data,
                                                                                         within_subject)

    cov_type, n_com = find_best_params(X_validate.to_numpy())
    gmm = GaussianMixture(covariance_type=cov_type, n_components=n_com)
    gmm.fit(X_train)
    optimal_cutoff, auc = roc_curving(gmm, X_validate, y_validate)
    TP, FP, TN, FN = prep_and_test(gmm, X_test, y_test, optimal_cutoff)
    acc = (TP + TN) / (TP + FP + TN + FN)

    return acc, auc


def run_multiple_gmm(dfs, mix_data, within_group):
    X_train, X_validate, X_test, y_train, y_validate, y_test = prep_train_test_data(dfs, mix_data, within_group)
    cov_type, n_com = find_best_params(X_validate)
    gmm = GaussianMixture(covariance_type=cov_type, n_components=n_com)
    gmm.fit(X_train)
    optimal_cutoff, auc = roc_curve_multiple(gmm, X_validate, y_validate)
    TP, FP, TN, FN = train_and_test_multiple(gmm, X_test, y_test, optimal_cutoff)
    acc = (TP + TN) / (TP + FP + TN + FN)

    return acc, auc