# -*- coding;utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn import metrics
from sklearn import preprocessing
# python的mrmr算法库存在问题
# import pymrmr
from feature_extract import get_feature


# 调用feature_extract提取有效特征，使用pymrmr进行特征选择，运行效果未达到预期，后续未使用
def mrmr():
    path = r'../AAdata/'
    for file_name in os.listdir(path):
        # if file_name != 'ALA.xlsx':
        #     continue
        # 读取数据，已清理后的数据，运行时许输入非热点和热点的阈值
        data = get_feature(path + file_name, float(sys.argv[1]), float(sys.argv[2]))
        for key in data.keys():
            print(key)
        # 将dict转化为DataFrame类型存储
        input = pd.DataFrame(data)
        if not os.path.exists(r'../cleaned_AA2data/'):
            os.makedirs(r'../cleaned_AA2data/')
        # 存储特征数据，mrmr_win32.exe只能读取csv文件，第一列必须为class，to_csv中的index=false即不添加行号
        input.to_csv(r'../cleaned_AA2data/' + file_name[0:3] + '.csv', index=False)
        # mRMR算法，python版本的mrmr使用效果不好，采用peng网站提供的exe文件进行mrmr算法
        # mrmr_reture = pymrmr.mRMR(input, 'MIQ', 20)
        # print(mrmr_reture)


mrmr()
