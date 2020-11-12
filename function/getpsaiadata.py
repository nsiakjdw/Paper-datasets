# -*- coding:utf-8 -*-

import os
import pandas as pd
import numpy as np
import re


def get_pdb(filename):
    pf = pd.read_excel(filename)
    # 存储数据信息
    list1 = []
    # 提取蛋白名，链，id和from的属性
    proteinName = list(pf.iloc[:, 0])
    proteinChain = list(pf.iloc[:, 2])
    proteinSequence = list(pf.iloc[:, 4])
    AA_from = list(pf.iloc[:, -1])
    # 只提取来源为single的数据，属性为single，则加入list1中
    for i in range(len(proteinName)):
        if AA_from[i] == 'single' and [proteinName[i], proteinChain[i], str(proteinSequence[i])] not in list1:
            list1.append([proteinName[i], proteinChain[i], str(proteinSequence[i])])
    return (list1)


def get_features_bound(dir, proteinInf):
    listdir = os.listdir(dir)

    # 1a4y与1a22顺序反了
    # item = listdir[0]
    # listdir[0] = listdir[1]
    # listdir[1] = item
    print('number of file:', len(listdir))
    # 存储特征信息
    listbound = []
    # 存储蛋白质名，链和id
    bound_information = []
    # 空格正则
    regex = re.compile('\s+')
    for filename in listdir:
        filestr = dir + filename
        # print(filestr, filename[0:4])
        # 读取文件信息
        lines = open(filestr)
        for line in lines:
            # 每行信息
            line = line[:-1]
            # 根据空格划分数据
            liststr = regex.split(line)
            # 去除首尾的空格
            liststr1 = liststr[1:-1]
            # 出现一种错误，liststr1[0] list index out of range,原因是某些行就是空的，不能引用
            # print(len(liststr1))
            # print(type(liststr1), len(liststr1))
            # print([filename[0:4].upper(), liststr[0]]) # , liststr[6]])
            # 行的划分为31的数据判断是否在skempi中，在的话分配到listbound中
            if len(liststr1) == 31:
                try:
                    if [filename[0:4].upper(), liststr1[0], liststr1[6]] in proteinInf:
                        # 保存列的信息
                        listbound.append([str for str in liststr1])
                        bound_information.append([filename[0:4].upper(), liststr1[0], liststr1[6]])
                        # print([filename[0:4].upper(), liststr1[0], liststr1[6]])
                except ValueError:
                    print('error')
    return listbound, bound_information


# 运行过程和bound是一样的
def get_features_unbound(dir, proteinInf):
    listdir = os.listdir(dir)
    print(len(listdir))
    # item = listdir[0]
    # listdir[0] = listdir[1]
    # listdir[1] = item

    listunbound = []
    regex = re.compile('\s+')
    for filename in listdir:
        filestr = dir + filename
        # print(filestr, filename[0:4])
        lines = open(filestr)
        for line in lines:
            line = line[:-1]
            liststr = regex.split(line)
            liststr = liststr[1:-1]
            if len(liststr) == 31:
                try:
                    if [filename[0:4].upper(), liststr[0], liststr[6]] in proteinInf:
                        # print('find')
                        listunbound.append([str.upper() for str in liststr])
                except ValueError:
                    pass
    return (listunbound)


# 提取from为single的数据求取重复的数据的平均值
def get_average_data(filename):
    # 查看单个文件运行是否正常
    # if filename != 'ARG.xlsx':
    #     continue
    # path末尾增加‘/’对检索文件没有关系，pandas读取文件需要完整路径，对path没有‘/’的增加path
    # if path[-1] != '/':
    #     filename = '/' + filename
    pf = pd.read_excel(filename)
    all_data = pf.iloc[:, :].values
    # 提取from为single的数据的序号
    tuple_of_discard_multi = np.where(all_data[:, -1] == 'single')
    print(type(tuple_of_discard_multi), tuple_of_discard_multi)
    # 提取single的数据保存在discard_multi中
    discard_multi = all_data[tuple_of_discard_multi[0]]
    # 联合pdb名，链和id名的数据
    union_sequence = discard_multi[:, 0] + discard_multi[:, 2] + [str(x) for x in discard_multi[:, 4]]
    # 创建dict统计数据，相同的key数据会存放到dict[key]中
    dict_statistic = {}
    for i in range(union_sequence.shape[0]):
        # 没有对应的key就创建
        if union_sequence[i] not in dict_statistic.keys():
            dict_statistic[union_sequence[i]] = []
        # 将数据存到对应的dict[key]中
        dict_statistic[union_sequence[i]].append(float(discard_multi[i][-2]))
    # 存储各个数据的平均值
    list_average_energy = []
    for key in dict_statistic.keys():
        # 将list转为array，使用array的求和函数计算list中数据的和，然后除以list的长度得到能量的平均值
        list_average_energy.append(np.array(dict_statistic[key]).sum() / len(dict_statistic[key]))
    # list_class = [0 if -0.4 < x < 0.4 else 1 for x in list_average_energy]
    return list_average_energy


def featureExcel(listbound, listunbound, energy, savefile):
    dictFeature = {}
    # bound和unbound的特征名
    name = ['chain_id', 'chain_total_ASA', 'chain_back_bone_ASA', 'chain_side_chain_ASA', 'chain_polar_ASA',
            'chain_no_polar_ASA', 'res_id', 'res_name', 'total_ASA', 'back_bone_ASA', 'side_chain_ASA',
            'polar_ASA', 'no_polar_ASA', 'total_RASA', 'back_bone_RASA', 'side_chain_RASA', 'polar_RASA',
            'no_polar_RASA', 'total_mean_DPX', 'total_standard_deviation_DPX', 'side_chain_mean_DPX',
            'side_chain_standard_deviation_DPX', 'maximum_DPX', 'minimum_DPX', 'total_mean_CX',
            'total_standard_deviation_CX', 'side_chain_mean_CX', 'side_chain_standard_deviation_CX',
            'maximum_CX', 'minimum_CX', 'hydrophobility']
    # 添加能量进入dictFeature中
    dictFeature['energy'] = energy
    # listbound中每项保存的是每个样本的信息，提取每个样本的1,2,3...个位置的数据形成list的数据类型存放到dict中相应的dict[key]中
    for j in range(31):
        dictFeature[name[j] + '_bound'] = ([listbound[i][j] for i in range(len(listbound))])
    for j in range(31):
        dictFeature[name[j] + '_unbound'] = ([listunbound[i][j] for i in range(len(listunbound))])
    print(len(dictFeature))
    # 保存数据
    save = pd.DataFrame(dictFeature)
    save.to_excel(savefile)


def main():
    # 读取各个单一文件
    path = r'../finaldata/'
    for filename in os.listdir(path):
        print(filename)
        # 单个文件测试效果
        # if filename != 'ALA.xlsx':
        #     continue
        # 返回skempi的数据，不包含重复的数据和from属性为multi的数据
        proteinInf = get_pdb(path + filename)
        # if filename == 'ARG.xlsx':
        #     print(proteinInf)
        print('length of proteinInf:', len(proteinInf))
        # bound的特征文件所在目录
        boundDir = r'../skempi2_psaia/bound/'
        # 返回单一的数据bound的特征和长度
        listbound, bound_information = get_features_bound(boundDir, proteinInf)
        print('length of listbound', len(listbound))
        # 查看未找全的pdb文件
        # for i in range(len(proteinInf)):
        #     if proteinInf[i] not in bound_information:
        #         print(proteinInf[i])
        # unbound的特征文件所在目录
        unboundDir = r'../skempi2_psaia/unbound/'
        listunbound = get_features_unbound(unboundDir, proteinInf)
        print('length of listunbound', len(listunbound))

        # 重复突变信息求取平均值,并将能量添加入原始数据中（舍弃multi的数据）
        energy = get_average_data(path + filename)
        print(len(energy))

        # 保存bound和unbound的原始数据信息
        savepath = r'../AA2data/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        featureExcel(listbound, listunbound, energy, savepath + filename)


if __name__ == "__main__":
    main()
