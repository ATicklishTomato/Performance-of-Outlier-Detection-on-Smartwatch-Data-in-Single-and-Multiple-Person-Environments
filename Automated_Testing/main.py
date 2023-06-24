import pickle
import random

import numpy as np

import preprocessing
import saving
import gmm
import dbscan
import os
from tqdm import tqdm

features = {
    'count': False,
    'mean': True,
    'std': True,
    'min': True,
    'percentiles': [0.25, 0.5, 0.75],
    'max': True,
    'n_th_step': True,
    'mfcc': True
}
using_feat = [x for x, y in features.items() if (type(y) == bool and y) or (type(y) == list and len(y) > 0)]

window_size = 3600 * 6  # in seconds
window_stride = 0  # in seconds

mixing = 0.1

subjects_path = './path/to/subject'
subject_directory_files = os.listdir(subjects_path)

# Data should be pickled Pandas DataFrames per subject consisting of timeseries data with columns 'time',
# 'hr' and 'steps'
all_subjects = [(subjects_path + x) if x.endswith('.pkl') else (subjects_path + (x + '.pkl')) for x in
                subject_directory_files]

preprocess_files = False
verbose_preprocessing = False
hr_feats = ['hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_0.25', 'hr_0.5', 'hr_0.75', 'hr_n_th_step', 'hr_mfcc']
steps_feats = ['steps_mean', 'steps_std', 'steps_min', 'steps_max', 'steps_0.25', 'steps_0.5', 'steps_0.75',
               'steps_n_th_step', 'steps_mfcc']


def run_test_situations(subjects='all', group_size=range(2, 7), number_of_groups_per_size=3):
    saving.initialize_files()

    if subjects == 'all':
        subjects = all_subjects
    else:
        subjects = [(subjects_path + x) if x.endswith('.pkl') else (subjects_path + (x + '.pkl')) for x in subjects]

    if preprocess_files:
        dfs, means_ranking = preprocess_subjects(subjects)
        for subject, df in dfs.items():
            with open(f'./preprocessed_windows/{subject.split("/")[-1]}', 'wb') as f:
                pickle.dump(df, f)
        with open(f'./preprocessed_windows/means_ranking.pkl', 'wb') as f:
            pickle.dump(means_ranking, f)
    else:
        dfs = {}
        for subject in tqdm(subjects, desc='Loading preprocessed subjects'):
            try:
                with open(f'./preprocessed_windows/{subject.split("/")[-1]}', 'rb') as f:
                    df = pickle.load(f)
                    # df = df.drop(hr_feats, axis=1)
                    dfs[subject] = df
            except Exception as e:
                if verbose_preprocessing:
                    print(f"Error in loading preprocessed subject {subject}: {e}")
        with open(f'./preprocessed_windows/means_ranking.pkl', 'rb') as f:
            means_ranking = pickle.load(f)

    perform_single_gmm(dfs)
    perform_multiple_gmm(dfs, group_size, number_of_groups_per_size)
    perform_single_dbscan(dfs)
    perform_multiple_dbscan(dfs, group_size, number_of_groups_per_size)

    saving.close_files()


def preprocess_subjects(subjects):
    subject_dfs = {}
    means_and_stds = {}
    for subject in tqdm(subjects, desc='Preprocessing subjects'):
        mean_std_data = None
        try:
            df, mean_std_data = preprocessing.sliding_window_creation([subject], window_size, features,
                                                                      window_stride=window_stride)
            subject_dfs[subject] = df[0]
        except Exception as e:
            subject_dfs[subject] = None
            print(f"Error in preprocessing subject {subject}: {e}")

        if mean_std_data is not None:
            savable_list = [subject] + list(mean_std_data.values())
            means_and_stds[subject] = mean_std_data
            saving.write_to_file('subject_mean_std', savable_list)
        else:
            means_and_stds[subject] = None
            saving.write_to_file('subject_mean_std', [subject, None, None, None, None])

    means_and_stds = {x: y for x, y in means_and_stds.items() if y is not None}
    means_ranking = [x for x, y in sorted(means_and_stds.items(), key=lambda item: item[1]['hr_mean'])]
    subjects_without_df = [x for x, y in subject_dfs.items() if y is None]
    subject_dfs = {x: y for x, y in subject_dfs.items() if y is not None}
    means_ranking = [x for x in means_ranking if x not in subjects_without_df]
    return subject_dfs, means_ranking


def perform_single_gmm(dfs):
    progress_bar = tqdm(total=len(dfs.keys()) * (len(dfs) - 1) * 2, desc="Performing single GMM")
    for subject, df in dfs.items():
        if df is None:
            continue
        mix_subjects = dfs.copy()
        mix_subjects.pop(subject)
        for mix_name, mix_data in mix_subjects.items():
            mix = mix_data.sample(n=min(len(mix_data.values), int(mixing * len(df.values))))
            try:
                acc, auc = gmm.run_single_gmm(df, mix, within_subject=True)
                saving.write_to_file('single_gmm', [subject, acc, auc, "within_subject", mix_name])
                progress_bar.update(1)
            except Exception as e:
                print(f"Error in performing single GMM, within subject, on subject {subject}: {e}")
                progress_bar.update(1)

            try:
                acc, auc = gmm.run_single_gmm(df, mix, within_subject=False)
                saving.write_to_file('single_gmm', [subject, acc, auc, "between_subjects", mix_name])
                progress_bar.update(1)
            except Exception as e:
                print(f"Error in performing single GMM, between subjects, on subject {subject}: {e}")
                progress_bar.update(1)
    progress_bar.close()


def perform_multiple_gmm(dfs, group_size, number_of_groups_per_size):
    # create different groups of dfs from group_size
    groups = []
    for size in group_size:
        for _ in range(0, number_of_groups_per_size):
            keys = random.sample(list(dfs.keys()), size)
            groups.append({x: dfs[x] for x in keys})

    iterations = np.sum((len(dfs.keys()) - np.array(list(group_size))) * number_of_groups_per_size)

    progress_bar = tqdm(total=iterations * 2, desc="Performing multiple GMM")
    for group in groups:
        names = group.keys()
        g_dfs = group.values()
        mix_subjects = dfs.copy()
        for name in names:
            mix_subjects.pop(name)

        for mix_name, mix in mix_subjects.items():
            mix = [mix.sample(n=min(len(mix.values), int(mixing * len(x.values)))) for x in g_dfs]
            try:
                acc, auc = gmm.run_multiple_gmm(g_dfs, mix, within_group=True)
                saving.write_to_file('multiple_gmm', [" ".join(names), acc, auc, "within_group", mix_name])
                progress_bar.update(1)
            except Exception as e:
                print(f"Error in performing multiple GMM, within group, on subject {names}: {e}")
                progress_bar.update(1)
            try:
                acc, auc = gmm.run_multiple_gmm(g_dfs, mix, within_group=False)
                saving.write_to_file('multiple_gmm',
                                     [" ".join(names), acc, auc, "between_group_and_individual", mix_name])
                progress_bar.update(1)
            except Exception as e:
                print(f"Error in performing multiple GMM, between group and subject, on subject {names}: {e}")
                progress_bar.update(1)
    progress_bar.close()


def perform_single_dbscan(dfs):
    feature_count = list(dfs.values())[0].shape[1]
    minPts = 2 * feature_count  # Sander et al., 1998

    progress_bar = tqdm(total=len(dfs.keys()) * (len(dfs) - 1) * 2, desc="Performing single DBSCAN")
    for subject, df in dfs.items():
        if df is None:
            continue
        mix_subjects = dfs.copy()
        mix_subjects.pop(subject)
        for mix_name, mix_data in mix_subjects.items():
            mix = mix_data.sample(n=min(len(mix_data.values), int(mixing * len(df.values))))
            try:
                acc, ari, ri, ss = dbscan.run_single_dbscan(df, minPts, mix, True)
                saving.write_to_file('single_dbscan', [subject, acc, ari, ri, ss, "within_subject", mix_name])
                progress_bar.update(1)
            except Exception as e:
                print(f"Error in performing single DBSCAN, within subject, on subject {subject}: {e}")
                progress_bar.update(1)
            try:
                acc, ari, ri, ss = dbscan.run_single_dbscan(df, minPts, mix, False)
                saving.write_to_file('single_dbscan', [subject, acc, ari, ri, ss, "between_subjects", mix_name])
                progress_bar.update(1)
            except Exception as e:
                print(f"Error in performing single DBSCAN, between subjects, on subject {subject}: {e}")
                progress_bar.update(1)
    progress_bar.close()


def perform_multiple_dbscan(dfs, group_size, number_of_groups_per_size):
    feature_count = list(dfs.values())[0].shape[1]
    minPts = 2 * feature_count  # Sander et al., 1998
    # create different groups of dfs from group_size
    groups = []
    for size in group_size:
        for _ in range(0, number_of_groups_per_size):
            keys = random.sample(list(dfs.keys()), size)
            groups.append({x: dfs[x] for x in keys})

    iterations = np.sum((len(dfs.keys()) - np.array(list(group_size))) * number_of_groups_per_size)

    progress_bar = tqdm(total=iterations * 2, desc="Performing multiple DBSCAN")
    for group in groups:
        names = group.keys()
        g_dfs = group.values()
        mix_subjects = dfs.copy()
        for name in names:
            mix_subjects.pop(name)

        for mix_name, mix in mix_subjects.items():
            mix = [mix.sample(n=min(len(mix.values), int(mixing * len(x.values)))) for x in g_dfs]
            try:
                acc, ari, ri, ss = dbscan.run_multiple_dbscan(g_dfs, minPts, mix, True)
                saving.write_to_file('multiple_dbscan', [" ".join(names), acc, ari, ri, ss, "within_group", mix_name])
                progress_bar.update(1)
            except Exception as e:
                print(f"Error in performing multiple within DBSCAN on subjects {names}: {e}")
                progress_bar.update(1)
            try:
                acc, ari, ri, ss = dbscan.run_multiple_dbscan(g_dfs, minPts, mix, False)
                saving.write_to_file('multiple_dbscan',
                                     [" ".join(names), acc, ari, ri, ss, "between_group_and_individual", mix_name])
                progress_bar.update(1)
            except Exception as e:
                print(f"Error in performing multiple between DBSCAN on subjects {names}: {e}")
                progress_bar.update(1)
    progress_bar.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_test_situations()
