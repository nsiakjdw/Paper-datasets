# 程序运行步骤
1.mutation.py提取所有数据，将多段数据拆分成单段数据存储
2.classifyeveryAminoAcid.py按照突变的氨基酸的种类分别放
3.getpsaiadata.py读取从psaia作用分析器提取的特征存储在AAdata目录，并将skempi中重复的数据进行平均处理（只提取来源为single的数据，舍弃multi的数据）
4.mrmr.py运行时，并不会使用pymrmr库的mrmr特征选择算法，只是使用feature_extract.py中的函数对特征进行处理，得到'RctASA', 'RcsASA', 'RctmPI', 'RcsmPI', 'RcpASA'，并去除一些无用的特征
pymrmr使用， 右键mrmr.py文件->open in terminal->输入 mrmr.py 0.4 2 ，选取-0.4到0.4之间的非热点和小于-2到大于2的热点
5.genearte_mrmr_sequence.py文件使用mrmr_win32.exe（peng个人网站下载的mrmr算法）的mrmr算法对数据进行特征选择。，结果存放在run_mrmr_ending目录
6.all_method_prediction.py使用mrmr特征选择的结果，使用svm算法对数据进行预测， 包含两个函数，一个是mrmr选择的结果，另一个是使用'RctASA', 'RcsASA', 'RctmPI', 'RcsmPI', 'RcpASA'作为特征得到的结果（未对所有数据进行预测）
选择的所有的数据集的预测结果是采用10折交叉结果中fscore最高的数据存放到predcted_AA目录