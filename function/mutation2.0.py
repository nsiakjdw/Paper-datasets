# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd

def main():
    # 文件名太长，只显示数据的相对位置所在的目录，os.listdir读取目录内的文件名
    path = r'../data/'
    for filename in os.listdir(path):
        if filename[-4:] != 'xlsx':
            continue
        # 提取原始数据，直接读取excel文件的内容，文件名需完整路径，即path + filename
        df = pd.read_excel(path + filename)
        # iloc函数提取全部数据，.values将数据转化为数组的形式
        data = df.iloc[:, :].values
        # 提取需要的列存储在矩阵当中
        matrix = data[:, [0, 1, 3, 10]]

        # 存储数据格式，保存在dict中，dict的结构{'Complex':[], 'originalResidue_name':[], 'chain':[], 'mutatedResidue_name':[], 'id':[], 'location':[], 'ddg':[], 'from':[]}
        # []为列表的形式，dict数据类型可以直接使用pandas库进行存储
        # 存储的数据的列的名字
        columnName = ['Complex', 'originalResidue_name', 'chain', 'mutatedResidue_name', 'id', 'location', 'ddg', 'from']
        dict = {}
        # 初始化dict
        for str in columnName:
            dict[str] = []
        # 氨基酸单字母对应的三字母氨基酸表示
        aminoAcid = {'A': 'ALA', 'G': 'GLY', 'L': 'LEU', 'I': 'ILE', 'V': 'VAL',
                     'P': 'PRO', 'F': 'PHE', 'M': 'MET', 'W': 'TRP', 'S': 'SER',
                     'Q': 'GLN', 'T': 'THR', 'C': 'CYS', 'N': 'ASN', 'Y': 'TYR',
                     'D': 'ASP', 'E': 'GLU', 'K': 'LYS', 'R': 'ARG', 'H': 'HIS'}
        # matrix.shape返回数组的大小，0的位置表示的矩阵的行数
        # range将int型数字转化为list列表形式
        for i in range(matrix.shape[0]):
            # 如果逗号不在某一行的第二列中，直接提取matrix中的数据保存在dict中
            if ',' not in matrix[i][1]:
                # 将第二列的第一个字母转化为氨基酸三个字母表示的形式保存在dict['originalResidue_name']中，append在[]中往后顺延添加
                dict[columnName[1]].append(aminoAcid[matrix[i][1][0]])
                # 将第二列的第二个字母链名存储在dict['chain']中
                dict[columnName[2]].append(matrix[i][1][1])
                # 将第二列的第三个数字到倒数第二个数字存储到dict['id']
                dict[columnName[4]].append(matrix[i][1][2:-1])
                # 将第二列的最后一个字母的转化为三字母的氨基酸表示， 保存在dict['mutatedResidue_name']
                dict[columnName[3]].append(aminoAcid[matrix[i][1][-1]])
                # 将第一列的数据的前四个字母保存在dict['complex']中
                dict[columnName[0]].append(matrix[i][0][0:4])
                # 将第3列的数据（location）保存在dict['location']中
                # matrix[i, 2]与matrix[i][2]表示的数据是相同的
                dict[columnName[5]].append(matrix[i, 2])
                # 将第4列的数据（ddg）保存在dict['ddg']中
                dict[columnName[6]].append(matrix[i, 3])
                # 将数据的来源保存到dict['from']中
                dict[columnName[7]].append('single')
            else:
                # 第二列和第三列对应的寻找到的可能突变的氨基酸，将两列数据分别根据逗号切片
                # 将matrix第一列切片的数据保存在strAA中，形式如['AA1D', 'AA2E']，第二列的location相同
                strAA = matrix[i][1].split(',')
                locationstr = matrix[i][2].split(',')
                # 将strAA的长度转为list列表形式
                for j in range(len(strAA)):
                    # 提取strAA的第j个的第一个字母转化为氨基酸三个字母表示的形式保存在dict['originalResidue_name']中
                    dict[columnName[1]].append(aminoAcid[strAA[j][0]])
                    # 提取strAA的第二个字母链名存储在dict['chain']中
                    dict[columnName[2]].append(strAA[j][1])
                    # 提取strAA的第三个数字到倒数第二个数字存储到dict['id']
                    dict[columnName[4]].append(strAA[j][2:-1])
                    # 提取最后一个字母的转化为三字母的氨基酸表示， 保存在dict['mutatedResidue_name']
                    dict[columnName[3]].append(aminoAcid[strAA[j][-1]])
                    # 与单数据相同
                    dict[columnName[0]].append(matrix[i][0][0:4])
                    dict[columnName[5]].append(locationstr[j])
                    dict[columnName[6]].append(matrix[i, 3])
                    dict[columnName[7]].append('multi')
        # 将dict存储在DataFrame中，可以直接存储在excel的一种形式
        df = pd.DataFrame(dict)
        # 保存数据的位置
        path = r'../tempdata/'
        # 如果不存在此条目录，创建目录
        if not os.path.exists(path):
            os.makedirs(path)
        # 将数据保存在path路径下的alldata文件中
        df.to_excel(path + 'alldata.xlsx')

if __name__ == '__main__':
    main()
