# -*- coding:utf-8 -*-

from __future__ import print_function
import math
import pprint
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from feature_add import generate_train_and_test_and_drop_categrical
from show_protein import generate
import matplotlib.pyplot as plt


def complete_file():
    df_result = pd.read_csv(r'../hot_region_dataset/hot_region_result.csv')
    test_sequence = generate_train_and_test_and_drop_categrical(r'../hot_region_dataset/cleaned_ALA_data.csv', label=False)
    df = pd.read_csv(r'../hot_region_dataset/cleaned_ALA_data.csv')
    all_data = df.iloc[:, 1:7].values
    test_data = all_data[test_sequence]
    df_result['pdb_name'] = test_data[:, 0]
    df_result['chain'] = test_data[:, 1]
    df_result['residue_id'] = test_data[:, 2]
    df_result['residue'] = test_data[:, 3]
    df_result['one_part_chain'] = test_data[:, 4]
    df_result['another_part_chain'] = test_data[:, 5]
    df_result.to_csv(r'../hot_region_dataset/hot_region_result.csv', index=None)


def complete_coordinate():
    df = pd.read_csv(r'../hot_region_dataset/hot_region_result.csv')
    residue = df.iloc[:, 7:10].values
    coordinate = []
    for i in range(residue.shape[0]):
        for line in open(r'../skempi2.0_pdb/' + residue[i, 0] + '.pdb').readlines():
            if line[:4] == 'ATOM' and line[21] == residue[i, 1] and line[22:27].split()[0] == residue[i, 2] and \
                    line[16] in [' ', 'A']:
                coordinate.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                break
    coordinate = np.array(coordinate)
    df['x'] = coordinate[:, 0]
    df['y'] = coordinate[:, 1]
    df['z'] = coordinate[:, 2]
    df.to_csv(r'../hot_region_dataset/hot_region_result.csv', index=None)


def get_hot_region_coordinate(label):
    df = pd.read_csv(r'../hot_region_dataset/hot_region_result.csv')
    data = df.iloc[:, :].values
    sequence = np.where(data[:, label] == 1)
    data = data[sequence]
    # print(data.shape)
    coordinate = {}
    for i in range(data.shape[0]):
        if data[i, 7] not in coordinate.keys():
            coordinate[data[i, 7]] = [[data[i, 8], data[i, 9], data[i, 10], data[i, -3], data[i, -2], data[i, -3]]]
        else:
            coordinate[data[i, 7]].append([data[i, 8], data[i, 9], data[i, 10], data[i, -3], data[i, -2], data[i, -3]])
    return coordinate


def get_standard_hot_region(label=0):
    coordinate = get_hot_region_coordinate(label)
    volumn = {'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1, 'CYS': 108.5, 'GLN': 143.8, 'GLU': 138.4,
              'GLY': 60.1, 'HIS': 153.2, 'ILE': 166.7, 'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9, 'PHE': 189.9,
              'PRO': 112.7, 'SER': 89, 'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140}
    hot_region = {}
    for key in coordinate.keys():
        matrix = np.zeros((len(coordinate[key]), len(coordinate[key])))
        for i in range(len(coordinate[key])):
            for j in range(len(coordinate[key])):
                real_distance = math.sqrt((float(coordinate[key][i][3]) - float(coordinate[key][j][3])) ** 2 +
                                          (float(coordinate[key][i][4]) - float(coordinate[key][j][4])) ** 2 +
                                          (float(coordinate[key][i][5]) - float(coordinate[key][j][5])) ** 2)
                hot_region_disatnce = math.pow(volumn[coordinate[key][i][2]] * 3 / 4 / math.pi, 1.0 / 3) + 2 + \
                                      math.pow(volumn[coordinate[key][j][2]] * 3 / 4 / math.pi, 1.0 / 3)
                if real_distance < hot_region_disatnce:
                    matrix[i][j] = 1

        #　print(key, matrix)
        region_divided = []
        list_visited = [0 for i in range(len(coordinate[key]))]
        num = 0

        while 0 in list_visited:
            single_region = []
            for i in range(len(list_visited)):
                if list_visited[i] == 0:
                    num = i
                    break
            queue = [num]
            while True:
                if len(queue) != 0:
                    if queue[0] not in single_region:
                        single_region.append(queue[0])
                    num = queue.pop(0)
                    list_visited[num] = 1
                    for i in range(len(coordinate[key])):
                        if matrix[num][i] == 1 and list_visited[i] == 0:
                            queue.append(i)
                else:
                    break
            region_divided.append(single_region)
        hot_region[key] = region_divided

    return hot_region, coordinate


# search all data of value from list_prediction
def search(list_prediction, value):
    list_find = []
    for i in range(len(list_prediction)):
        if list_prediction[i] == value:
            list_find.append(i)
    return list_find


# predict by dbscan
def DBSCAN_predict(label, distance, ms):
    """
    :param label: predicted by methosd
    :param distance: DBSCAN parameter distamce
    :param ms: DBSCAN parameter minimum samples
    :return: predicted hot region and details of hot region
    """
    coordinate = get_hot_region_coordinate(label)
    predicted_hot_region = {}
    data = {}

    # extract coordinate of amino acids from data, and save into np.array()
    for key in coordinate.keys():
        data[key] = np.array(coordinate[key])[:, -3:]
        data[key] = data[key].astype(np.float32)
    for key in data.keys():
        predicted_hot_region[key] = []
        # -1 presents it's not in cluster, multi cluster will be 0, 1, 2, 3...
        predicted_label = DBSCAN(eps=distance, min_samples=ms).fit_predict(np.array(data[key]))
        # search all cluster
        for i in range(len(predicted_label)):
            if i in predicted_label:
                predicted_hot_region[key].append(search(predicted_label, i))
            else:
                break
    return predicted_hot_region, coordinate


def is_predicted_hot_region(predicted_hot_region, standard_hot_region, score):
    right_hot_spots = 0
    for i in range(len(predicted_hot_region)):
        if predicted_hot_region[i] in standard_hot_region:
            right_hot_spots += 1
    if right_hot_spots/len(standard_hot_region) >= score:
        return True, right_hot_spots
    return False, right_hot_spots


def main():
    # complete data
    complete_file()
    complete_coordinate()
    s_hot_region, s_coordinate = get_standard_hot_region(label=0)
    # print(s_hot_region)
    # print(len(s_hot_region))
    number_of_hot_region = 0
    for key in s_hot_region.keys():
        for i in range(len(s_hot_region[key])):
            if len(s_hot_region[key][i]) >= 3:
                number_of_hot_region += 1
    # print(number_of_hot_region)

    # hot region in details
    all_standard_hot_region = {}
    for key in s_hot_region.keys():
        for i in range(len(s_hot_region[key])):
            if len(s_hot_region[key][i]) >= 3:
                s_hot_region[key][i].sort()
                if key not in all_standard_hot_region.keys():
                    all_standard_hot_region[key] = []
                hot_region = []
                for j in range(len(s_hot_region[key][i])):
                    hot_region.append(s_coordinate[key][s_hot_region[key][i][j]])
                all_standard_hot_region[key].append(hot_region)
    dict_r, dict_p, dict_f = {}, {}, {}
    svm, xgboost, rf, ann, gnb = 0, 0, 0, 0, 0
    for m in range(6):
        matrix_r, matrix_p, matrix_f = np.zeros((20, 10)), np.zeros((20, 10)), np.zeros((20, 10))
        for n in range(20):
            for k in range(10):
                p_hot_region, p_coordinate = DBSCAN_predict(label=m+1, distance=(n+1)/2.0, ms=k)
                # print(p_hot_region)
                # print(len(p_hot_region))
                # print(number_of_hot_region)

                number_of_predicted_hot_region = 0
                for key in p_hot_region.keys():
                    for i in range(len(p_hot_region[key])):
                        if len(p_hot_region) >= 3:
                            number_of_predicted_hot_region += 1
                # print(number_of_predicted_hot_region)
                all_predicted_hot_region = {}
                for key in p_hot_region.keys():
                    for i in range(len(p_hot_region[key])):
                        if len(p_hot_region[key][i]) >= 3:
                            p_hot_region[key][i].sort()
                            if key not in all_predicted_hot_region.keys():
                                all_predicted_hot_region[key] = []
                            hot_region = []
                            for j in range(len(p_hot_region[key][i])):
                                hot_region.append(p_coordinate[key][p_hot_region[key][i][j]])
                            all_predicted_hot_region[key].append(hot_region)

                right_hot_region = 0
                # measure of predicted hot region
                for key in all_predicted_hot_region.keys():
                    for i in range(len(all_predicted_hot_region[key])):
                        if key in all_standard_hot_region.keys():
                            for j in range(len(all_standard_hot_region[key])):
                                state, number = is_predicted_hot_region(all_predicted_hot_region[key][i],
                                                                        all_standard_hot_region[key][j], 0.6)
                                if state:
                                    right_hot_region += 1
                # print("***********************")
                # print(number_of_hot_region, number_of_predicted_hot_region, right_hot_region)
                recall = right_hot_region/number_of_hot_region
                precision = right_hot_region/number_of_predicted_hot_region if number_of_predicted_hot_region != 0 else 0
                f_score = 2 * recall * precision / (recall + precision) if recall+precision != 0 else 0
                # print('methods:{0}, distance:{1}, min_sampels:{2}, recall:{3}, precision:{4}, f-score:{5}'
                #       .format(m, n, k, recall, precision, f_score))
                matrix_r[n][k], matrix_p[n][k], matrix_f[n][k] = recall, precision, f_score
                if m == 3 and n == 29 and k == 3:
                    pprint.pprint(all_standard_hot_region)
                    # pprint.pprint()
                    print('hot region:  standard and prediction')
                    for key in all_standard_hot_region.keys():
                        print(key, end=': ')
                        if key not in all_predicted_hot_region.keys():
                            for i in range(len(all_standard_hot_region[key])):
                                print('the {0} hot region:'.format(i+1), end=': ')
                                for j in range(len(all_standard_hot_region[key][i])):
                                    print(''.join(all_standard_hot_region[key][i][j][0:3]), end=', ')
                            print('--------', end='')
                            print('none-----0')
                        else:
                            for i in range(len(all_standard_hot_region[key])):
                                print('the {0} hot region:'.format(i+1), end=': ')
                                for j in range(len(all_standard_hot_region[key][i])):
                                    print(''.join(all_standard_hot_region[key][i][j][0:3]), end=', ')
                            print('--------', end='')
                            for i in range(len(all_predicted_hot_region[key])):
                                print('the {0} hot region:'.format(i+1), end=': ')
                                for j in range(len(all_predicted_hot_region[key][i])):
                                    print(''.join(all_predicted_hot_region[key][i][j][0:3]), end=', ')
                            print()
                            # right_hot_spots = 0
                            # for i in range(len(all_predicted_hot_region[key])):
                            #     for j in range(len(p_hot_region[key][i])):
                            #         if all_predicted_hot_region[i] in all_standard_hot_region[key]:
                            #             right_hot_spots += 1
                elif m == 4 and n == 17 and k == 4:
                    gnb = all_predicted_hot_region
                elif m == 0 and n == 19 and k == 3:
                    svm = all_predicted_hot_region
                elif m == 1 and n==19 and k ==3:
                    xgboost = all_predicted_hot_region
                elif m == 2 and n == 19 and k == 3:
                    rf = all_predicted_hot_region
                elif m == 3 and n == 19 and k ==3:
                    ann = all_predicted_hot_region
        dict_r[m], dict_p[m], dict_f[m] = matrix_r, matrix_p, matrix_f
    # for key in dict_r.keys():
    #     for i in range(dict_r[key].shape[0]):
    #         for j in range(dict_r[key].shape[1]):
    #             print(key, i, j, dict_r[key][i][j], dict_p[key][i][j], dict_f[key][i][j])
    print([dict_f[key].max() for key in dict_f.keys()])
    print([np.where(dict_f[key] == dict_f[key].max()) for key in dict_f.keys()])
    pprint.pprint(svm.keys())
    pprint.pprint(xgboost.keys())
    pprint.pprint(rf.keys())
    pprint.pprint(ann.keys())
    pprint.pprint(gnb.keys())
    pprint.pprint(all_standard_hot_region.keys())
    pprint.pprint(svm['1AO7'])
    pprint.pprint(xgboost['1AO7'])
    pprint.pprint(rf['1AO7'])
    pprint.pprint(ann['1AO7'])
    pprint.pprint(gnb['1AO7'])
    pprint.pprint(all_standard_hot_region['1AO7'])
    for key in all_standard_hot_region.keys():
        if key in gnb.keys():
            generate(key, all_standard_hot_region[key], gnb[key], r'../pymol_pml/gnb/')
    for key in all_standard_hot_region.keys():
        if key in ann.keys():
            generate(key, all_standard_hot_region[key], ann[key], r'../pymol_pml/ann/')
    for key in all_standard_hot_region.keys():
        if key in rf.keys():
            generate(key, all_standard_hot_region[key], rf[key], r'../pymol_pml/rf/')
    for key in all_standard_hot_region.keys():
        if key in xgboost.keys():
            generate(key, all_standard_hot_region[key], xgboost[key], r'../pymol_pml/xgb/')
    for key in all_standard_hot_region.keys():
        if key in svm.keys():
            generate(key, all_standard_hot_region[key], svm[key], r'../pymol_pml/svm/')

    print('只有gnb预测出来的结果：')
    for key in gnb.keys():
        if key in all_standard_hot_region.keys() and key not in svm.keys() and key not in xgboost.keys() \
                and key not in rf.keys() and key not in ann.keys():
            print(key)

    right_svm, right_xgboost, right_rf, right_ann, right_gnb = [], [], [], [], []
    hotspot_svm, hotspot_xgboost, hotspot_rf, hotspot_ann, hotspot_gnb = {}, {}, {}, {}, {}

    for key in all_standard_hot_region.keys():
        for i in range(len(all_standard_hot_region[key])):
            hotspot_svm[key+str(i)] = []
            hotspot_xgboost[key+str(i)] = []
            hotspot_rf[key+str(i)] = []
            hotspot_ann[key+str(i)] = []
            hotspot_gnb[key+str(i)] = []
            if key in svm.keys():
                for j in range(len(svm[key])):
                    state, num = is_predicted_hot_region(svm[key][j], all_standard_hot_region[key][i], 0.6)
                    hotspot_svm[key+str(i)].append(num)
            else:
                hotspot_svm[key + str(i)].append(0)

            if key in xgboost.keys():
                for j in range(len(xgboost[key])):
                    state, num = is_predicted_hot_region(xgboost[key][j], all_standard_hot_region[key][i], 0.6)
                    hotspot_xgboost[key+str(i)].append(num)
            else:
                hotspot_xgboost[key + str(i)].append(0)

            if key in rf.keys():
                for j in range(len(rf[key])):
                    state, num = is_predicted_hot_region(rf[key][j], all_standard_hot_region[key][i], 0.6)
                    hotspot_rf[key+str(i)].append(num)
            else:
                hotspot_rf[key + str(i)].append(0)

            if key in ann.keys():
                for j in range(len(ann[key])):
                    state, num = is_predicted_hot_region(ann[key][j], all_standard_hot_region[key][i], 0.6)
                    hotspot_ann[key+str(i)].append(num)
            else:
                hotspot_ann[key + str(i)].append(0)

            if key in gnb.keys():
                for j in range(len(gnb[key])):
                    state, num = is_predicted_hot_region(gnb[key][j], all_standard_hot_region[key][i], 0.6)
                    hotspot_gnb[key+str(i)].append(num)
            else:
                hotspot_gnb[key + str(i)].append(0)

    for key in svm.keys():
        for i in range(len(svm[key])):
            if key in all_standard_hot_region.keys():
                hotspot_svm[key] = []
                for j in range(len(all_standard_hot_region[key])):
                    state, num = is_predicted_hot_region(svm[key][i], all_standard_hot_region[key][j], 0.6)
                    if state:
                        right_svm.append(key)
                    # hotspot_svm[key].append(num)
    for key in xgboost.keys():
        for i in range(len(xgboost[key])):
            if key in all_standard_hot_region.keys():
                hotspot_xgboost[key] = []
                for j in range(len(all_standard_hot_region[key])):
                    state, num = is_predicted_hot_region(xgboost[key][i], all_standard_hot_region[key][j], 0.6)
                    if state:
                        right_xgboost.append(key)
                    # hotspot_xgboost[key].append(num)
    for key in rf.keys():
        for i in range(len(rf[key])):
            if key in all_standard_hot_region.keys():
                hotspot_rf[key] = []
                for j in range(len(all_standard_hot_region[key])):
                    state, num = is_predicted_hot_region(rf[key][i], all_standard_hot_region[key][j], 0.6)
                    if state:
                        right_rf.append(key)
                    # hotspot_rf[key].append(num)
    for key in ann.keys():
        for i in range(len(ann[key])):
            if key in all_standard_hot_region.keys():
                hotspot_ann[key] = []
                for j in range(len(all_standard_hot_region[key])):
                    state, num = is_predicted_hot_region(ann[key][i], all_standard_hot_region[key][j], 0.6)
                    if state:
                        right_ann.append(key)
                    # hotspot_ann[key].append(num)
    for key in gnb.keys():
        for i in range(len(gnb[key])):
            if key in all_standard_hot_region.keys():
                hotspot_gnb[key] = []
                for j in range(len(all_standard_hot_region[key])):
                    state, num = is_predicted_hot_region(gnb[key][i], all_standard_hot_region[key][j], 0.6)
                    if state:
                        right_gnb.append(key)
                    # hotspot_gnb[key].append(num)
    print(len(right_svm), len(right_xgboost), len(right_rf), len(right_ann), len(right_gnb))
    print()
    aa = []
    aa.extend(right_svm)
    aa.extend(right_xgboost)
    aa.extend(right_rf)
    aa.extend(right_ann)
    aa.extend(right_gnb)
    aa = np.array(aa)
    aa = np.unique(aa)
    print("除GNB外全部预测错误的热区")
    print(aa)
    for key in aa:
        if key in right_svm:
            print(key, end=' ')
        else:
            print(None, end=' ')
        if key in right_xgboost:
            print(key, end=' ')
        else:
            print(None, end=' ')
        if key in right_rf:
            print(key, end=' ')
        else:
            print(None, end=' ')
        if key in right_ann:
            print(key, end=' ')
        else:
            print(None, end=' ')
        if key in right_gnb:
            print(key, end=' ')
        else:
            print(None, end=' ')
        if key in all_standard_hot_region.keys():
            print(key)
        else:
            print(None)

    print('svm:')
    for key in svm.keys():
        print(key, end='  ')
        for i in range(len(svm[key])):
            for j in range(len(svm[key][i])):
                print(''.join(svm[key][i][j][0:3]), end=' ')
            print()
    print('xgboost:')
    for key in xgboost.keys():
        print(key, end='  ')
        for i in range(len(xgboost[key])):
            for j in range(len(xgboost[key][i])):
                print(''.join(xgboost[key][i][j][0:3]), end=' ')
            print()
    print('rf:')
    for key in rf.keys():
        print(key, end='  ')
        for i in range(len(rf[key])):
            for j in range(len(rf[key][i])):
                print(''.join(rf[key][i][j][0:3]), end=' ')
            print()
    print('ann:')
    for key in ann.keys():
        print(key, end='  ')
        for i in range(len(ann[key])):
            for j in range(len(ann[key][i])):
                print(''.join(ann[key][i][j][0:3]), end=' ')
            print()
    print('gnb:')
    for key in gnb.keys():
        print(key, end='  ')
        for i in range(len(gnb[key])):
            for j in range(len(gnb[key][i])):
                print(''.join(gnb[key][i][j][0:3]), end=' ')
            print()
    print('standard hot region:')
    for key in all_standard_hot_region.keys():
        print(key, end='  ')
        for i in range(len(all_standard_hot_region[key])):
            for j in range(len(all_standard_hot_region[key][i])):
                print(''.join(all_standard_hot_region[key][i][j][0:3]), end=' ')
            print()

    svm_plt, xgboost_plt, rf_plt, ann_plt, gnb_plt, shr_plt, label = [], [], [], [], [], [], []
    for key in all_standard_hot_region.keys():
        for i in range(len(all_standard_hot_region[key])):
            print(max(hotspot_svm[key+str(i)]), max(hotspot_xgboost[key+str(i)]), max(hotspot_rf[key+str(i)]),
                  max(hotspot_ann[key+str(i)]), max(hotspot_gnb[key+str(i)]), len(all_standard_hot_region[key][i]))
            svm_plt.append(max(hotspot_svm[key+str(i)]))
            xgboost_plt.append(max(hotspot_xgboost[key+str(i)]))
            rf_plt.append(max(hotspot_rf[key+str(i)]))
            ann_plt.append(max(hotspot_ann[key+str(i)]))
            gnb_plt.append(max(hotspot_gnb[key+str(i)]))
            shr_plt.append(len(all_standard_hot_region[key][i]))
            if len(all_standard_hot_region[key]) > 1:
                label.append(key+str(i))
            else:
                label.append(key)
    # plt.figure()
    # plt.plot(np.linspace(0, len(svm_plt), len(svm_plt)), svm_plt, label='svm')
    # plt.plot(np.linspace(0, len(svm_plt), len(svm_plt)), xgboost_plt, label='xgboost')
    # plt.plot(np.linspace(0, len(svm_plt), len(svm_plt)), rf_plt, label='random forest')
    # plt.plot(np.linspace(0, len(svm_plt), len(svm_plt)), ann_plt, label='artificial neural network')
    # plt.plot(np.linspace(0, len(svm_plt), len(svm_plt)), gnb_plt, label='gaussian naive bayes')
    # plt.plot(np.linspace(0, len(svm_plt), len(svm_plt)), shr_plt, label='standard hot region')
    # plt.xlabel(label)
    # plt.legend()
    # plt.show()
    ind = np.arange(len(label))
    width = 0.15
    fig, ax = plt.subplots()# dpi = 800)
    # plt.tick_params(labelsize=4)
    plt.style.use('seaborn-dark')
    rect = []
    rect.append(ax.bar(ind - width / 2 * 5, shr_plt, width, label='Hot region'))
    rect.append(ax.bar(ind - width / 2 * 3, gnb_plt, width, label='gnb'))
    rect.append(ax.bar(ind - width / 2, svm_plt, width, label='svm'))
    rect.append(ax.bar(ind + width / 2, xgboost_plt, width, label='xgboost'))
    rect.append(ax.bar(ind + width / 2 * 3, rf_plt, width, label='random forest'))
    rect.append(ax.bar(ind + width / 2 * 5, ann_plt, width, label='ann'))
    ax.set_ylabel('Number',fontsize=15)
    ax.set_title('Distribution of hot spots in hot regions')
    # ind = [value + 1 for value in ind]
    ax.set_xticks(ind)
    ax.legend()

    # def autolabel(rects):
    #     for r in rects:
    #         height = r.get_height()
    #         ax.text(r.get_x()+r.get_width()/2.0, 1.01 * height, '{}'.format(height), ha='center', va='bottom', fontsize=4)
    # for i in range(len(rect)):
    #     autolabel(rect[i])
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax.legend( fontsize=15)
    # ax.set_ylabel(..., fontsize=15)
    # plt.savefig(r'hot_region_distribution.png', dpi=800)
    plt.show()


if __name__ == '__main__':
    main()
