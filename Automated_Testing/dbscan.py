import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score
from kneebow.rotor import Rotor


def single_find_epsilon(df, minPts):
    """Find best epsilon using the elbow curve of K-NN distances and its point of maximum curviture"""
    neighbors = NearestNeighbors(n_neighbors=minPts)
    neighbors_fit = neighbors.fit(df)
    distances, indices = neighbors_fit.kneighbors(df)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    coords = np.column_stack((range(0,len(distances)), distances))
    rotor = Rotor()
    rotor.fit_rotate(coords)
    elbow_index = rotor.get_elbow_index()
    epsilon = coords[elbow_index][1]

    return epsilon

def single_select_outliers(df, mixing_data, within_subject):
    if within_subject:
        numpy_data = df.to_numpy()
        window_lengths = [np.linalg.norm(x) for x in numpy_data]
        outlier_indexes = np.argpartition(window_lengths, (-1 * int(0.1 * len(numpy_data))))[
                          (-1 * int(0.1 * len(numpy_data))):]
        y_true = np.ones(len(numpy_data))
        y_true[outlier_indexes] = 0
        df['class'] = y_true
        mixing_data['class'] = np.ones(len(mixing_data.values))
        df = pd.concat([df, mixing_data], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)  # shuffle df
        y_true = df['class'].tolist()
        df = df.drop(['class'], axis=1)
    else:
        df['class'] = np.ones(len(df.values))
        mixing_data['class'] = np.zeros(len(mixing_data.values))
        df = pd.concat([df, mixing_data], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)  # shuffle df
        y_true = df['class'].tolist()
        df = df.drop(['class'], axis=1)
    return df, y_true

def multiple_select_outliers(dfs, mixing_data, within_group):
    if within_group:
        y_trues = []
        new_dfs = []
        for df in dfs:
            numpy_data = df.to_numpy()
            window_lengths = [np.linalg.norm(x) for x in numpy_data]
            outlier_indexes = np.argpartition(window_lengths, (-1 * int(0.1 * len(numpy_data))))[
                              (-1 * int(0.1 * len(numpy_data))):]
            y_true = np.ones(len(numpy_data))
            y_true[outlier_indexes] = 0
            df['class'] = y_true
            new_dfs.append(df)

        df = pd.concat(new_dfs, ignore_index=True)
        y_true = df['class'].tolist()
        df = df.drop(['class'], axis=1)
    else:
        df = pd.concat(dfs, ignore_index=True)
        df['class'] = np.ones(len(df.values))
        mix = pd.concat(mixing_data, ignore_index=True)
        mix['class'] = np.zeros(len(mix.values))
        df = pd.concat([df, mix], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)  # shuffle df
        y_true = df['class'].tolist()
        df = df.drop(['class'], axis=1)
    return df, y_true

def multiple_find_epsilon(df, minPts, within_group):
    """Find best epsilon using the elbow curve of K-NN distances and its point of maximum curviture"""
    neighbors = NearestNeighbors(n_neighbors=minPts)
    neighbors_fit = neighbors.fit(df)
    distances, indices = neighbors_fit.kneighbors(df)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    coords = np.column_stack((range(0,len(distances)), distances))
    rotor = Rotor()
    rotor.fit_rotate(coords)
    elbow_index = rotor.get_elbow_index()
    epsilon = coords[elbow_index][1]
    return epsilon

def run_single_dbscan(df, minPts, mixing_data, within_subject):
    df, y_true = single_select_outliers(df, mixing_data, within_subject)
    epsilon = single_find_epsilon(df, minPts)
    dbscan = DBSCAN(eps=epsilon, min_samples=minPts)
    predictions = dbscan.fit_predict(df)

    cleaned_predictions = [0 if x == -1 else 1 for x in predictions]
    TP = [1 for x, y in zip(cleaned_predictions, y_true) if x == 0 and y == 0]
    FP = [1 for x, y in zip(cleaned_predictions, y_true) if x == 0 and y == 1]
    TN = [1 for x, y in zip(cleaned_predictions, y_true) if x == 1 and y == 1]
    FN = [1 for x, y in zip(cleaned_predictions, y_true) if x == 1 and y == 0]

    acc = (len(TP) + len(TN)) / (len(TP) + len(FP) + len(TN) + len(FN))
    ari = adjusted_rand_score(y_true, cleaned_predictions)
    ri = rand_score(y_true, cleaned_predictions)
    ss = silhouette_score(df, cleaned_predictions)

    return acc, ari, ri, ss

def run_multiple_dbscan(dfs, minPts, mixing_data, within_group):
    df, y_true = multiple_select_outliers(dfs, mixing_data, within_group)
    epsilon = multiple_find_epsilon(df, minPts, within_group)
    dbscan = DBSCAN(eps=epsilon, min_samples=minPts)
    predictions = dbscan.fit_predict(df)
    # print([epsilon] + list(set(predictions)))
    cleaned_predictions = [0 if x == -1 else 1 for x in predictions]
    TP = [1 for x, y in zip(cleaned_predictions, y_true) if x == 0 and y == 0]
    FP = [1 for x, y in zip(cleaned_predictions, y_true) if x == 0 and y == 1]
    TN = [1 for x, y in zip(cleaned_predictions, y_true) if x == 1 and y == 1]
    FN = [1 for x, y in zip(cleaned_predictions, y_true) if x == 1 and y == 0]

    acc = (len(TP) + len(TN)) / (len(TP) + len(FP) + len(TN) + len(FN))
    ari = adjusted_rand_score(y_true, cleaned_predictions)
    ri = rand_score(y_true, cleaned_predictions)
    ss = silhouette_score(df, cleaned_predictions)

    return acc, ari, ri, ss
