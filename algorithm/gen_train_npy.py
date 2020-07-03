# -*- coding: utf-8 -*-
"""
Created on Thu Jun  11 19:04:12 2020

@author: zhanghanxiao
"""
#-------------------------dependencies--------------------------
import pandas as pd
import numpy as np
import time
import math

#---------------------------const value -----------------------------
road_name = ['NanPing_W2E', 'NanPing_E2W', 'FuLong_S2N', 'FuLong_N2S', 'LiuXian_W2E', 'LiuXian_E2W',
             'YuLong_S2N', 'YuLong_N2S', 'XinQu_S2N', 'XinQu_N2S', 'ZhiYuan_S2N', 'ZhiYuan_N2S']
assert(len(road_name)==12)

dict_road_id = {276183:'NanPing_W2E', 276184:'NanPing_E2W', 275911:'FuLong_S2N',  275912:'FuLong_N2S',
                276240:'LiuXian_W2E', 276241:'LiuXian_E2W', 276264:'YuLong_S2N',  276265:'YuLong_N2S',
                276268:'XinQu_S2N',   276269:'XinQu_N2S',   276737:'ZhiYuan_S2N', 276738:'ZhiYuan_N2S'
                }  # map road names to the road ids

dict_road_index = {276183: 0, 276184: 1, 275911: 2,  275912: 3,
                   276240: 4, 276241: 5, 276264: 6,  276265: 7,
                   276268: 8, 276269: 9, 276737: 10, 276738: 11
                   }

#--------------------generate train array-----------------------
def gen_train(train_data):
    feature = []
    label = [[], [], []]
    for row in range(6, train_data.shape[0]-2):
        tmp = []
        for i in range(6,0,-1):
            tmp.append((train_data.iloc[row-i][0] % 86400) / 600) # related to time
            tmp.append(train_data.iloc[row-i][1])  # TTI
            tmp.append(train_data.iloc[row-i][2])  # num
            tmp.append(train_data.iloc[row-i][3])  # speed
        feature.append(tmp)
        label[0].append(train_data.iloc[row][1])
        label[1].append(train_data.iloc[row+1][1])
        label[2].append(train_data.iloc[row+2][1])
    feature = np.array(feature)
    label = np.array(label)
    return feature, label

def gen_test(test_data):
    feature = []
    for row in range(6, train_data.shape[0]-2):
        tmp = []
        for i in range(6,0,-1):
            tmp.append((test_data.iloc[row-i][0] % 86400) / 600) # related to time
            tmp.append(test_data.iloc[row-i][1])  # TTI
            tmp.append(test_data.iloc[row-i][2])  # num
            tmp.append(test_data.iloc[row-i][3])  # speed
        feature.append(tmp)
    feature = np.array(feature)
    return feature

def main():
    for i in range(len(road_name)):
        train_data = pd.read_csv("../datasets/_"+road_name[i]+".csv", sep=',')
        train_data = train_data.sort_values(by = 'timestamp')
        X, y = gen_train(train_data)
        X_filename = 'test_array/X_12_' + road_name[i] + '.npy'
        np.save(X_filename, X)
        y_filename = 'test_array/y_12_' + road_name[i] + '.npy'
        np.save(y_filename, y)
        print("save train: ", i)

if __name__ == "__main__":
    main()