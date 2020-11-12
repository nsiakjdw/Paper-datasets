# -*- coding:utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise, BatchNormalization, GaussianDropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras import losses
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import load_model
from keras import initializers

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import scale
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge

from xgboost import XGBClassifier, XGBRegressor

from roc_curve_skempi import generate_roc_curve


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
    file_name = r'E:/python_engineer/skempi2.0_experiment/hot_region_dataset/ALA.txt'

    # 执行exe程序
    os.system(r'E:/python_engineer/skempi2.0_experiment/mrmr_win32.exe -i ' +
              r'E:/python_engineer/skempi2.0_experiment/hot_region_dataset/cleaned_ALA_data.csv'
              + ' -m MIQ -n 83 -t 1 >' + file_name)


def predicted_by_svm(train_x, train_y, test_x, test_y, sequence):
    list_result = []
    list_fscore = []
    train_y = [0 if value <= 1 else 1 for value in train_y]
    proba = 0
    for i in range(83):
        clf = SVC(kernel='linear', C=1, probability=True)
        # clf = SVR(kernel='rbf', C=2, gamma=0.1)
        clf.fit(scale(train_x[:, sequence[:i + 1]]), train_y)
        predicted = clf.predict(scale(test_x[:, sequence[:i + 1]]))
        print(accuracy_score(test_y, predicted))
        if i == 4:
            proba = clf.predict_proba(scale(test_x[:, sequence[:i + 1]]))
        # clf.fit(scale(train_x[:, sequence[:i + 1]]), train_y)
        # pred = clf.predict(scale(test_x[:, sequence[:i + 1]]))
        # predicted = [0 if pred[i] <= 1 else 1 for i in range(len(pred))]
        precision, recall, fscore, _ = precision_recall_fscore_support(test_y, predicted, labels=[1])
        print('svm: features:{0}, recall:{1}, precision:{2}, f-score:{3}'.format(i, recall, precision, fscore))
        list_result.append(predicted)
        list_fscore.append(fscore)
    return list_result[list_fscore.index(max(list_fscore))], proba


def predicted_by_rf(train_x, train_y, test_x, test_y, sequence):
    list_result = []
    list_fscore = []
    train_y = [0 if value <= 1 else 1 for value in train_y]
    proba = 0
    for i in range(83):
        clf = RandomForestClassifier(n_estimators=200, max_depth=3, n_jobs=4, random_state=100, max_features=2 if i > 2 else 1)
        # clf = RandomForestRegressor(n_estimators=200, max_depth=3, n_jobs=4, random_state=100, criterion='rmsd')
        clf.fit(train_x[:, sequence[:i + 1]], train_y)
        predicted = clf.predict(test_x[:, sequence[:i + 1]])
        if i == 1:
            proba = clf.predict_proba(test_x[:, sequence[:i + 1]])
        print(accuracy_score(test_y, predicted))
        # pred = clf.predict(scale(test_x[:, sequence[:i + 1]]))
        # predicted = [0 if pred[i] <= 1 else 1 for i in range(len(pred))]
        precision, recall, fscore, _ = precision_recall_fscore_support(test_y, predicted, labels=[1])
        print('Random Forest: features:{0}, recall:{1}, precision:{2}, all_f-score:{3}'
              .format(i, recall, precision, fscore))
        list_result.append(predicted)
        list_fscore.append(fscore)
    return list_result[list_fscore.index(max(list_fscore))], proba


def predicted_by_xgb(train_x, train_y, test_x, test_y, sequence):
    list_result = []
    list_fscore = []
    train_y = [0 if value <= 1 else 1 for value in train_y]
    proba = 0
    for i in range(65):
        clf = XGBClassifier(n_estimators=200, max_depth=3, n_jobs=4, random_state=i + 1, learning_rate=0.05,
                            subsample=0.8, min_child_weight=1, reg_alpha=1e-4, gamma=0.1)
        # clf = XGBRegressor(n_estimators=200, max_depth=3, n_jobs=4, random_state=i + 1, learning_rate=0.05,
        #                    subsample=0.7, min_child_weight=1, reg_alpha=1e-4, gamma=0.1)
        clf.fit(train_x[:, sequence[:i + 1]], train_y)
        # print(clf.score(train_x[:, sequence[:i + 1]], train_y), clf.score(test_x[:, sequence[:i + 1]], test_y))
        predicted = clf.predict(test_x[:, sequence[:i + 1]])
        print(accuracy_score(test_y, predicted))
        if i == 30:
            proba = clf.predict_proba(test_x[:, sequence[:i + 1]])
        # pred = clf.predict(test_x[:, sequence[:i + 1]])
        # predicted = [0 if pred[i] <= 1 else 1 for i in range(len(pred))]
        precision, recall, fscore, _ = precision_recall_fscore_support(test_y, predicted, labels=[1])
        print('xgboost: features:{0}, recall:{1}, precision:{2}, f-score:{3}'.format(i, recall, precision, fscore))
        list_result.append(predicted)
        list_fscore.append(fscore)
    return list_result[list_fscore.index(max(list_fscore))], proba


def baseline_model():
    global num
    model = Sequential()
    model.add(Dense(input_dim=83, kernel_initializer=initializers.uniform(seed=0), bias_initializer=initializers.zeros(), activation='relu',
                    units=30))
    # model.add(BatchNormalization())
    model.add(GaussianNoise(1))
    model.add(GaussianDropout(0.3))
    # model.add(Dense(20, kernel_initializer='uniform', bias_initializer='uniform', activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(10, kernel_initializer=initializers.uniform(seed=0), bias_initializer='zeros', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, kernel_initializer=initializers.uniform(seed=0), bias_initializer='zeros', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, kernel_initializer=initializers.uniform(seed=0), bias_initializer='zeros', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, kernel_initializer=initializers.uniform(seed=0), bias_initializer='zeros', activation='softmax'))
    # Compile model
    model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    return model


def baseline_model_load(number):
    model = load_model(r'../model/model' + str(number) + '.h5')
    return model


def add_one_dimension(list_y):
    y = np.zeros((len(list_y), 2))
    for i in range(len(list_y)):
        if list_y[i] == 0:
            y[i][0], y[i][1] = 1, 0
        else:
            y[i][0], y[i][1] = 0, 1
    return y


def predicted_by_nn(train_x, train_y, test_x, test_y):
    list_result = []
    list_fscore = []
    pre, re, fs, num = 0, 0, 0, 0
    train_y = [0 if value <= 1 else 1 for value in train_y]
    proba = 0
    if not os.path.exists(r'../model'):
        os.mkdir(r'../model/')
        for j in range(10):
            # using Keras Classifier
            # estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=32, verbose=0)
            # c_train_y = add_one_dimension(train_y)
            # estimator.fit(scale(train_x), c_train_y)
            # pred = estimator.predict(scale(test_x))
            # print(estimator.score(scale(test_x), np_utils.to_categorical(test_y, num_classes=2)))
            # print(test_y, '\n', pred)

            # model predict
            model = baseline_model()
            c_train_y = add_one_dimension(train_y)
            model.fit(x=scale(train_x), y=c_train_y, epochs=50, batch_size=32, verbose=0)
            pred_y = model.predict(scale(test_x))
            pred = [0 if pred_y[i][0] > pred_y[i][1] else 1 for i in range(pred_y.shape[0])]
            print(accuracy_score(test_y, pred))
            print(pred_y)
            model.save(r'../model/save'+str(j)+'.h5')
            precision, recall, fscore, _ = precision_recall_fscore_support(test_y, pred, labels=[1])
            list_result.append(pred)
            list_fscore.append(fscore)
            print(recall, precision, fscore)
            pre += precision
            re += recall
            fs += fscore
            if fscore != 0:
                num += 1
        print(re/num, pre/num, 2*re*pre/(re+pre)/num)
    else:
        for j in range(10):
            model = load_model(r'../model/save'+str(j)+'.h5')
            pred_y = model.predict(scale(test_x))
            if j == 8:
                proba = pred_y
            pred = [0 if pred_y[i][0] > pred_y[i][1] else 1 for i in range(pred_y.shape[0])]
            print(accuracy_score(test_y, pred))
            precision, recall, fscore, _ = precision_recall_fscore_support(test_y, pred, labels=[1])
            print(recall, precision, fscore)
            list_result.append(pred)
            list_fscore.append(fscore)
            pre += precision
            re += recall
            fs += fscore
            if fscore != 0:
                num += 1
        print(re / num, pre / num, 2 * re * pre / (re + pre) / num)
    return list_result[8], proba


def predicted_by_naive_bayes(train_x, train_y, test_x, test_y, sequence):
    list_result = []
    list_fscore = []
    train_y = [0 if value <= 1 else 1 for value in train_y]
    proba = 0
    for i in range(83):
        gnb = GaussianNB()
        # gnb = BernoulliNB()
        gnb.fit(scale(train_x[:, sequence[:i + 1]]), train_y)
        print(gnb.class_prior_)
        pred = gnb.predict(scale(test_x[:, sequence[:i + 1]]))
        if i == 62:
            proba = gnb.predict_proba(scale(test_x[:, sequence[:i + 1]]))
        print(accuracy_score(test_y, pred))
        precision, recall, fscore, _ = precision_recall_fscore_support(test_y, pred, labels=[1])
        print('gaussian nb: features:{0}, recall:{1}, precision:{2}, f-score:{3}'.format(i, recall, precision, fscore))
        list_result.append(pred)
        list_fscore.append(fscore)
    return list_result[list_fscore.index(max(list_fscore))], proba


# def predicted_by_naive_bayes(train_x, train_y, test_x, test_y, sequence):
#     list_result = []
#     list_fscore = []
#     # train_y = [0 if value <= 1 else 1 for value in train_y]
#     for i in range(83):
#         gnb = GaussianNB()
#         # gnb = BernoulliNB()
#         gnb.fit(scale(train_x[:, sequence[:i + 1]]), train_y)
#         print(gnb.class_prior_)
#         pred = gnb.predict(scale(test_x[:, sequence[:i + 1]]))
#         precision, recall, fscore, _ = precision_recall_fscore_support(test_y, pred, labels=[1])
#         print('gaussian nb: features:{0}, recall:{1}, precision:{2}, f-score:{3}'.format(i, recall, precision, fscore))
#         list_result.append(pred)
#         list_fscore.append(fscore)
#     return list_result[list_fscore.index(max(list_fscore))]


def predicted_by_decision_tree(train_x, train_y, test_x, test_y, sequence):
    list_result = []
    list_fscore = []
    train_y = [0 if value <= 1 else 1 for value in train_y]
    for i in range(83):
        dt = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=i, max_features='sqrt', max_depth=3)
        # dt = DecisionTreeRegressor(criterion='mse', splitter='best', random_state=i, max_features='sqrt', max_depth=3)
        dt.fit(train_x[:, sequence[:i + 1]], train_y)
        predicted = dt.predict(test_x[:, sequence[:i + 1]])
        print(accuracy_score(test_y, predicted))
        # pred = dt.predict(test_x[:, sequence[:i + 1]])
        # predicted = [0 if pred[i] <= 1 else 1 for i in range(len(pred))]
        precision, recall, fscore, _ = precision_recall_fscore_support(test_y, predicted, labels=[1])
        print('decision tree: features:{0}, recall:{1}, precision:{2}, f-score:{3}'.format(i, recall, precision, fscore))
        list_result.append(predicted)
        list_fscore.append(fscore)
    return list_result[list_fscore.index(max(list_fscore))]


def main(parameter):
    df_train = pd.read_csv(r'../hot_region_dataset/dataset/train.csv')
    df_test = pd.read_csv(r'../   /dataset/test.csv')
    train = df_train.iloc[:, :].values
    test = df_test.iloc[:, :].values
    train = np.nan_to_num(train)
    test = np.nan_to_num(test)
    train_x, train_y = train[:, 1:], train[:, 0]
    test_x, test_y = test[:, 1:], test[:, 0]
    # train_y = [0 if value <= 1 else 1 for value in train_y]
    test_y = [0 if value <= 1 else 1 for value in test_y]
    print(train_x.shape, test_x.shape)

    execute_mrmr()
    sequence = get_rank_of_features()
    print(sequence)
    df = pd.DataFrame()
    df['true'] = test_y
    p = []

    if 'svm' in parameter:
        result1, proba1 = predicted_by_svm(train_x, train_y, test_x, test_y, sequence)
        df['svm'] = result1
        # print(proba1.shape, proba1)
        p.append(proba1)
    if 'xgb' in parameter:
        result2, proba2 = predicted_by_xgb(train_x, train_y, test_x, test_y, sequence)
        df['xgb'] = result2
        p.append(proba2)
        # print(proba2.shape, proba2)
    if 'rf' in parameter:
        result3, proba3 = predicted_by_rf(train_x, train_y, test_x, test_y, sequence)
        df['rf'] = result3
        p.append(proba3)
    if 'nn' in parameter:
        result4, proba4 = predicted_by_nn(train_x, train_y, test_x, test_y)
        df['nn'] = result4
        p.append(proba4)
    if 'nb' in parameter:
        result5, proba5 = predicted_by_naive_bayes(train_x, train_y, test_x, test_y, sequence)
        df['nb'] = result5
        print(proba5)
        p.append(proba5)
    if 'dt' in parameter:
        result6 = predicted_by_decision_tree(train_x, train_y, test_x, test_y, sequence)
        df['dt'] = result6
    df.to_csv(r'../hot_region_dataset/hot_region_result.csv', index=None)
    generate_roc_curve(test_y, p)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main(parameter=['svm', 'xgb', 'rf', 'nn', 'nb', 'dt'])
