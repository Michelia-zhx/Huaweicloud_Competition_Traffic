# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:32:25 2020

@author: zhanghanxiao
"""
#---------------dependencies--------------------------
import pandas as pd
import numpy as np
import time
import math

#---------------const value --------------------------
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

# ---------------load the data------------------------
processed_data_file = ["../processed_train_data/pro_20191201_20191220.csv", "../processed_train_data/pro_201910_11.csv", 
                       "../processed_train_data/pro_201901_201903.csv", "../processed_test_data/pro_toPredict_gps.csv"]
file_index = 3
processed_data = pd.read_csv(processed_data_file[file_index], sep=',')
processed_data.columns = ['timestamp', 'NanPing_W2E', 'NanPing_E2W', 'FuLong_S2N', 'FuLong_N2S', 'LiuXian_W2E', 'LiuXian_E2W',
                          'YuLong_S2N', 'YuLong_N2S', 'XinQu_S2N', 'XinQu_N2S', 'ZhiYuan_S2N', 'ZhiYuan_N2S']

processed_data = processed_data.sort_values('timestamp').reset_index().drop('index', axis=1)
data_len = len(processed_data)
processed_data = processed_data.set_index('timestamp')

# ---------------generate training set and test set------------------------
# a sample should contain road_id, timestamp, car_count, TTI, spead
label = pd.read_csv("../traffic/toPredict_train_TTI.csv").set_index('time')
lst_lst = []

for i in range(len(road_name)):
    lst_lst.append([])

for index, row in label.iterrows():
    time_str = index
    timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(timeArray)
    road_id = row['id_road']
    speed = row['speed']
    tti = row['TTI']
    try:
        data_row = processed_data.loc[timestamp]
        car_count = data_row[dict_road_id[road_id]]
    except:
        continue
    sample = np.array([timestamp, tti, car_count, speed])
    lst_lst[dict_road_index[road_id]].append(sample)

for i in range(len(road_name)):
    sample_mat = np.array(lst_lst[i])
    sample_df = pd.DataFrame(sample_mat)
    sample_df.columns = ['timestamp', 'TTI', 'car_count', 'speed']
    sample_df = sample_df.set_index('timestamp')
    print(sample_df)
    file_name = "../datasets/test_"+str(road_name[i])+".csv"
    pd.DataFrame.to_csv(sample_df, file_name, sep=',')