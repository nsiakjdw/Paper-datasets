# -*- coding:utf-8 -*-

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier


aa = ['ALA', 'ARG', 'ASP', 'ASN', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
      'LEU', 'LYS', 'PRO', 'SER', 'MET', 'THR', 'TRP', 'TYR', 'PHE', 'VAL']


def dataset_split(parameter, threshold):
    """
    split averagely all data into train data and test data in different amino acids according to the residue is bond
    by other residue
    :return:
    """
    path = r'../source_data/cleaned_ALA_data.csv'
    df = pd.read_csv(path)
    data = df.iloc[:, :].values
    train_data = None
    test_data = None
    if parameter == 0:
        for value in aa:
            aa_data = data[np.where(data[:, 4] == value)]
            nonhot = aa_data[np.where(aa_data[:, 0] <= threshold[0])]
            hot = aa_data[np.where(aa_data[:, 0] >= threshold[1])]
            nonhot_train, nonhot_test = train_test_split(nonhot, test_size=0.3, random_state=200)
            hot_train, hot_test = train_test_split(hot, test_size=0.3, random_state=200)
            if train_data is None:
                train_data = np.vstack((nonhot_train, hot_train))
                test_data = np.vstack((nonhot_test, hot_test))
            else:
                train_data = np.vstack((train_data, nonhot_train))
                train_data = np.vstack((train_data, hot_train))
                test_data = np.vstack((test_data, nonhot_test))
                test_data = np.vstack((test_data, hot_test))
    elif parameter == 1:
        for value in aa:
            aa_data = data[np.where(data[:, 4] == value)]
            nonhot = aa_data[np.where(aa_data[:, 0] <= threshold)]
            hot = aa_data[np.where(aa_data[:, 0] >= threshold)]
            nonhot_train, nonhot_test = train_test_split(nonhot, test_size=0.3, random_state=200)
            hot_train, hot_test = train_test_split(hot, test_size=0.3, random_state=200)
            if train_data is None:
                train_data = np.vstack((nonhot_train, hot_train))
                test_data = np.vstack((nonhot_test, hot_test))
            else:
                train_data = np.vstack((train_data, nonhot_train))
                train_data = np.vstack((train_data, hot_train))
                test_data = np.vstack((test_data, nonhot_test))
                test_data = np.vstack((test_data, hot_test))
    if parameter == 2:
        data = data[np.where(data[:, -1] > 0)]
        for value in aa:
            aa_data = data[np.where(data[:, 4] == value)]
            nonhot = aa_data[np.where(aa_data[:, 0] <= threshold)]
            hot = aa_data[np.where(aa_data[:, 0] >= threshold)]
            nonhot_train, nonhot_test = train_test_split(nonhot, test_size=0.3, random_state=200)
            hot_train, hot_test = train_test_split(hot, test_size=0.3, random_state=200)
            if train_data is None:
                train_data = np.vstack((nonhot_train, hot_train))
                test_data = np.vstack((nonhot_test, hot_test))
            else:
                train_data = np.vstack((train_data, nonhot_train))
                train_data = np.vstack((train_data, hot_train))
                test_data = np.vstack((test_data, nonhot_test))
                test_data = np.vstack((test_data, hot_test))
    pd_train = pd.DataFrame(train_data, columns=df.columns)
    pd_test = pd.DataFrame(test_data, columns=df.columns)
    pd_train.to_csv(r'../source_data/train.csv', index=None)
    pd_test.to_csv(r'../source_data/test.csv', index=None)


def cleaned_data(threshold):
    df_train = pd.read_csv(r'../source_data/train.csv')
    df_test = pd.read_csv(r'../source_data/test.csv')
    df_train.drop(df_train.columns[1:7], axis=1, inplace=True)
    df_test.drop(df_test.columns[1:7], axis=1, inplace=True)
    if isinstance(threshold, list):
        df_train['energy'][df_train['energy'] <= threshold[0]] = 0
        df_train['energy'][df_train['energy'] >= threshold[1]] = 1
        df_test['energy'][df_test['energy'] <= threshold[0]] = 0
        df_test['energy'][df_test['energy'] >= threshold[1]] = 1
    elif isinstance(threshold, int):
        df_train['energy'][df_train['energy'] < threshold] = 0
        df_train['energy'][df_train['energy'] >= threshold] = 1
        df_test['energy'][df_test['energy'] < threshold] = 0
        df_test['energy'][df_test['energy'] >= threshold] = 1
    if not os.path.exists(r'../dataset/'):
        os.makedirs(r'../dataset/')
    train = df_train.iloc[:, :].values
    print(train.shape)
    test = df_test.iloc[:, :].values
    df_all_data = pd.DataFrame(np.vstack((train, test)), columns=df_train.columns)
    df_all_data.to_csv(r'../dataset/all_data.csv', index=None)
    df_train.to_csv(r'../dataset/train.csv', index=None)
    df_test.to_csv(r'../dataset/test.csv', index=None)


# calculate the order of features
def get_rank_of_features():
    flag = 0
    sequence = []
    for line in open(r'../dataset/ALA.txt').readlines():
        if len(line[:-1].split()) == 4 and line[:-1].split()[1] == 'mRMR':
            flag = 1
        elif flag == 1 and len(line[:-1].split()) == 4 and line[:-1].split()[0].isdigit():
            sequence.append(int(line[:-1].split()[1])-1)
    return sequence


def execute_mrmr():
    # 执行mrm_win32.exe生成特征选择的结果
    file_name = r'E:/python_engineer/skempi2.0_experiment/dataset/ALA.txt'

    # 执行exe程序
    os.system(r'E:/python_engineer/skempi2.0_experiment/mrmr_win32.exe -i ' +
              r'E:/python_engineer/skempi2.0_experiment/dataset/train.csv' + ' -m MIQ -n 64 -t 1 >' + file_name)


def main(parameter, method):
    # ml = ['svm', 'xgboost', 'random forest', 'logisitic', 'bayesis', 'ANN']
    file = None
    if not os.path.exists(r'../result/'):
        os.makedirs(r'../result/')
    if parameter == 0:
        dataset_split(parameter, [0.4, 2])
        cleaned_data([0.4, 2])
        file = open('../result/result_' + method + '_0.4-2.txt', 'w')
    elif parameter == 1:
        dataset_split(parameter, 1)
        cleaned_data(1)
        file = open('../result/result_' + method + '_1.txt', 'w')
    elif parameter == 2:
        dataset_split(parameter, 2)
        cleaned_data(2)
        file = open('../result/result_' + method + '_2.txt', 'w')

    execute_mrmr()
    sequence = get_rank_of_features()

    df_train = pd.read_csv(r'../dataset/train.csv')
    train = df_train.iloc[:, :].values
    x_train = train[:, 1:]
    x_train = np.nan_to_num(x_train)
    y_train = train[:, 0]
    df_test = pd.read_csv(r'../dataset/test.csv')
    test = df_test.iloc[:, :].values
    x_test = test[:, 1:]
    x_test = np.nan_to_num(x_test)
    y_test = test[:, 0]
    for i in range(64):
        if method == 'svm':
            param = None
            if parameter == 0:
                param = {'kernel': 'rbf', 'C': 20}
            elif parameter == 1:
                param = {'kernel': 'rbf', 'C': 1.0}
            elif parameter == 2:
                param = {'kernel': 'rbf', 'C': 50}
            clf = SVC(kernel='linear', C=2)
            clf.fit(preprocessing.scale(x_train[:, sequence[:i+1]]), y_train)
            predicted = clf.predict(preprocessing.scale(x_test[:, sequence[:i+1]]))
            precision, recall, fscore, _ = precision_recall_fscore_support(y_test, predicted, labels=[1])
            print('svm: features:{0}, recall:{1}, precision:{2}, f-score:{3}'.format(i, recall, precision, fscore))
            file.writelines('svm: features:{0}, recall:{1}, precision:{2}, f-score:{3}'.format(i, recall, precision, fscore))
        elif method == 'xgb':
            pre, re, fs = 0, 0, 0
            for j in range(100):
                clf = XGBClassifier(n_estimators=150, max_depth=3, n_jobs=4, random_state=j+1)
                clf.fit(x_train[:, sequence[:i + 1]], y_train)
                predicted = clf.predict(x_test[:, sequence[:i + 1]])
                precision, recall, fscore, _ = precision_recall_fscore_support(y_test, predicted, labels=[1])
                pre += precision
                re += recall
                fs += fscore
            print('XGBoost: features:{0}, recall:{1}, precision:{2}, average_f-score:{3}, all_f-score:{4}'.format(
                i, re / 100, pre / 100, fs / 100, 2 * re * pre / (re + pre) / 100))
            file.writelines('XGBoost: features:{0}, recall:{1}, precision:{2}, average_f-score:{3}, '
                            'all_f-score:{4}'.format(i, re / 100, pre / 100, fs / 100, 2 * re * pre / (re + pre) / 100))
        elif method == 'rf':
            pre, re, fs = 0, 0, 0
            for j in range(100):
                clf = RandomForestClassifier(n_estimators=300, max_depth=5, n_jobs=4, min_samples_split=3, random_state=j)
                clf.fit(x_train[:, sequence[:i+1]], y_train)
                predicted = clf.predict(x_test[:, sequence[:i+1]])
                precision, recall, fscore, _ = precision_recall_fscore_support(y_test, predicted, labels=[1])
                pre += precision
                re += recall
                fs += fscore
            print('Random Forest: features:{0}, recall:{1}, precision:{2}, average_f-score:{3}, all_f-score:{4}'.format(
                i, re/100, pre/100, fs/100, 2*re*pre/(re+pre)/100))
            file.writelines('Random Forest: features:{0}, recall:{1}, precision:{2}, average_f-score:{3}, '
                            'all_f-score:{4}'.format(i, re/100, pre/100, fs/100, 2*re*pre/(re+pre)/100))
    file.close()


if __name__ == '__main__':
    """
    parameter = 0, 1, 2
    0 represents 0.4 and 2
    1 represents 1
    2 represents 2
    """
    warnings.filterwarnings('ignore')
    arg = argparse.ArgumentParser()
    arg.add_argument('--method', type=str)
    arg.add_argument('--split', type=float)
    args = arg.parse_args()

    try:
        if args.split not in [0, 1, 2]:
            raise Exception('do not have the split')
        if args.method not in ['svm', 'xgb', 'rf']:
            raise Exception('not support the method')
    except Exception as err:
        print(err)

    main(parameter=args.split, method=args.method)
