import pandas as pd
import numpy as np
import scipy as sc
import librosa as lib
from sklearn.preprocessing import StandardScaler
import random
import pickle
import os

import warnings
warnings.filterwarnings("ignore")


def sliding_window_creation(subjects, window_size, features, window_stride=0, mixing=0,
                            subjects_path='./fallback/path', subject_directory_files=None):
    """Create sliding windows based on parameters in setup block above."""
    if subject_directory_files is None:
        subject_directory_files = os.listdir(subjects_path)

    # Load data
    subject_data = []
    for filename in subjects:
        with open(filename, 'rb') as file:
            unpickled = pickle.load(file)
            if len(unpickled.index) < 100 * (window_size - window_stride):
                raise Exception("subject must have at least enough data for 100 windows, can only make " + \
                                f"{len(unpickled.index) / (window_size - window_stride)} for subject {filename}.")
            unpickled, mean_std_data = filter_data(unpickled, filename)
            subject_data.append(unpickled)

    # Create windows of data
    windowed_subject_data = []
    for subject in subject_data:
        windowed_subject_data.append(create_windows(subject, window_size, features, window_stride=window_stride))

    # Create mix data and add to windowed subject data
    if mixing > 0:
        # create set of files not yet used
        potential_mix_subjects = list(set([subjects_path + x for x in subject_directory_files]) - set(subjects))
        # pick a random one
        mix_subject = random.choice(potential_mix_subjects)
        # extract data and window it
        with open(mix_subject, 'rb') as file:
            unpickled = pickle.load(file)
            unpickled = filter_data(unpickled, "MIX: " + mix_subject)
            mix_data_window = create_windows(unpickled)
            mix_data = []
            # create random mix sample for every subject data set based on desired fraction
            # or entire mix_data if |mix_data| < mixing*|windowed_subject|
            for subject in windowed_subject_data:
                mix_data.append(mix_data_window.sample(n=min(len(mix_data_window.values),
                                                             int(mixing * len(subject.values)))))

    if mixing > 0:
        return windowed_subject_data, mean_std_data, mix_data
    else:
        return windowed_subject_data, mean_std_data


def normalize(window):
    mean = window.mean(skipna=True)
    std = window.std(ddof=0, skipna=True)
    window = (window - mean) / std
    return window


def filter_data(data, filename, hr=True, steps=True, norm_hr=True, norm_steps=True):
    # Interpolate missing data
    data['hr'] = data['hr'].ffill().bfill()
    data['steps'] = data['steps'].ffill().bfill()
    data['steps'] = data['steps'].sparse.to_dense()

    # Record mean & std
    mean_std_data = {
        "hr_mean": data['hr'].mean(skipna=True),
        "hr_std": data['hr'].std(ddof=0, skipna=True),
        "steps_mean": data['steps'].mean(skipna=True),
        "steps_std": data['steps'].std(ddof=0, skipna=True),
    }

    # Normalize if needed
    if norm_hr:
        data['hr'] = normalize(data['hr'])
    if norm_steps:
        data['steps'] = normalize(data['steps'])

    data["elapsed_time"] = (data["time"] - data["time"][0]).dt.total_seconds()

    # Select desired data
    if hr and steps:
        return data, mean_std_data
    elif hr:
        return data.drop(['steps'], axis=1), mean_std_data
    elif steps:
        return data.drop(['hr'], axis=1), mean_std_data
    else:
        raise ValueError("Can't make windows with no data, dummy")


def describe(window, start_time, end_time, features, n_th_step_moment=6, prefix=""):
    description = {}
    if features['count']:
        description[prefix + 'count'] = len(window)
    if features['mean']:
        description[prefix + 'mean'] = window.mean(skipna=True)
    if features['std']:
        description[prefix + 'std'] = window.std(ddof=0, skipna=True)
    if features['min']:
        description[prefix + 'min'] = window.min()
    if len(features['percentiles']) > 0:
        for quant in features['percentiles']:
            description[prefix + str(quant)] = window.quantile(quant)
    if features['max']:
        description[prefix + 'max'] = window.max()
    if features['n_th_step']:
        description[prefix + 'n_th_step'] = sc.stats.moment(window, moment=n_th_step_moment, nan_policy='omit')
    if features['mfcc']:
        time = (end_time - start_time)
        sr = len(window.values) / time
        res = lib.feature.mfcc(y=window.to_numpy(), sr=sr, n_mfcc=1, n_mels=1)[0]
        description[prefix + 'mfcc'] = np.mean(res)

    return description


def create_windows(subject_data, window_size, features, window_stride=0, hr=True, steps=True):
    index = 0
    data = []
    window_end_index = index + window_size
    new_start_index = window_end_index - window_stride
    while new_start_index < subject_data['elapsed_time'].values[-1]:
        # Create, describe, and store windows
        new_window = subject_data[subject_data['elapsed_time'].between(index, window_end_index, inclusive='left')]
        if len(new_window.values) == 0:
            index = new_start_index
            window_end_index = index + window_size
            new_start_index = window_end_index - window_stride
            continue
        new_row = {}
        if hr and steps:
            hr_window = new_window['hr']
            new_row.update(describe(hr_window, index, window_end_index, features=features, prefix="hr_"))
            steps_window = new_window['steps']
            new_row.update(describe(steps_window, index, window_end_index, features=features, prefix="steps_"))
        elif hr:
            hr_window = new_window['hr']
            new_row.update(describe(hr_window, index, window_end_index, features=features, prefix="hr_"))
        elif steps:
            steps_window = new_window['steps']
            new_row.update(describe(steps_window, index, window_end_index, features=features, prefix="steps_"))

        data.append(pd.Series(new_row))

        # Prepare for next loop
        index = new_start_index
        window_end_index = index + window_size
        new_start_index = window_end_index - window_stride

    return pd.DataFrame(data, columns=new_row.keys())


def train_without_outliers(normal_data, outlier_data, addition_normal_to_test):
    normalizer = StandardScaler()
    normalizer.set_output(transform='pandas')

    # Create new test data and drop it from the original data
    additional_test_data = normal_data.sample(n=addition_normal_to_test)
    normal_data = pd.concat([normal_data, additional_test_data]).drop_duplicates(keep=False)

    # Add class indication, combine, and shuffle
    additional_test_data['class'] = 1
    outlier_data['class'] = 0
    test_data = pd.concat([additional_test_data, outlier_data])
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    normal_data = normalizer.fit_transform(normal_data)
    test_classes = test_data['class']
    test_data = normalizer.transform(test_data.drop(['class'], axis=1))

    return normal_data, test_data, np.ones(len(normal_data)), test_classes
