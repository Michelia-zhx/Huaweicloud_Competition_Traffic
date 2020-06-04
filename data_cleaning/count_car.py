# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 08:49:25 2020

@author: 98061
"""
#---------------dependencies----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import math

#---------------const value ------------
road_name = ['NanPing_W2E', 'NanPing_E2W', 'FuLong_S2N', 'FuLong_N2S', 'LiuXian_W2E', 'LiuXian_E2W',
             'YuLong_S2N', 'YuLong_N2S', 'XinQu_S2N', 'XinQu_N2S', 'ZhiYuan_S2N', 'ZhiYuan_N2S']
assert(len(road_name)==12)
dict_road_pos = {'NanPing_W2E':(114.015136,22.587219,90),   'NanPing_E2W':(114.015136,22.587219,270),
                 'FuLong_S2N':(114.015666,22.603308,0),     'FuLong_N2S':(114.015666,22.603308,180),
                 'LiuXian_W2E':(114.027811,22.612045,90),   'LiuXian_E2W':(114.027811,22.612045,270),
                 'YuLong_S2N':(114.027826,22.605467,0),     'YuLong_N2S':(114.027826,22.605467,180),
                 'XinQu_S2N':(114.03428,22.604796,0),       'XinQu_N2S':(114.03428,22.604796,180),
                 'ZhiYuan_S2N':(114.025425,22.608589,0),    'ZhiYuan_N2S':(114.025425,22.608589,180)
                 }  # map raod names to the position and direction
dict_road_id = {'NanPing_W2E':276183, 'NanPing_E2W':276184, 'FuLong_S2N':275911,  'FuLong_N2S':275912,
                'LiuXian_W2E':276240, 'LiuXian_E2W':276241, 'YuLong_S2N':276264,  'YuLong_N2S':276265,
                'XinQu_S2N':276268,   'XinQu_N2S':276269,   'ZhiYuan_S2N':276737, 'ZhiYuan_N2S':276728
                 }  # map road names to the road ids

#--------------count from GPS--------
'''
number of cars passing the roads
columns: NanPing is a road name, W2E means from west to east
rows: per 30 min, index is start timestamp
'''

car_count = pd.DataFrame()
car_count['NanPing_W2E'] = None
car_count['NanPing_E2W'] = None
car_count['FuLong_S2N']  = None
car_count['FuLong_N2S']  = None
car_count['LiuXian_W2E'] = None
car_count['LiuXian_E2W'] = None
car_count['YuLong_S2N']  = None
car_count['YuLong_N2S']  = None
car_count['XinQu_S2N']   = None
car_count['XinQu_N2S']   = None
car_count['ZhiYuan_S2N'] = None
car_count['ZhiYuan_N2S'] = None

# -----------------function-------------
def cal_dist(pos_jing,pos_wei,road_x,road_y):
    return math.sqrt((pos_jing-road_x)**2 + (pos_wei-road_y)**2)

# ---------------load the data-----------------
read_file_name = ["../../20191201_20191220.csv", "../../201910_11.csv", "../../201901_201903.csv", "../../traffic/toPredict_train_gps.csv"]
save_file_name = ["../processed_train_data/pro_20191201_20191220.csv", "../processed_train_data/pro_201910_11.csv", 
                  "../processed_train_data/pro_201901_201903.csv", "../processed_test_data/pro_toPredict_gps.csv"]
file_index = 3
read_file = read_file_name[file_index]
save_file = save_file_name[file_index]
raw_data = pd.read_csv(read_file, iterator = True);
chunk_size  = 1000
round_count = 0
while(True):
    round_count += 1
    try:
        gps = raw_data.get_chunk(chunk_size)
        print(round_count)
        # continue
    except:
        break
    gps.columns = ["order","user","info"]

    # ----------------clean the data-------------------
    # add time info as a column
    timestamp_lst = []  # timestamp
    time_read     = []  # readable time
    for row in range(gps.shape[0]):
        data = gps.iloc[row][2].strip("[").strip("]")
        gps.iloc[row][2] = data
        timestamp = int(data.split(",")[0].split(" ")[4])
        timeArray = time.localtime(timestamp)
        otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
        timestamp_lst.append(timestamp)
        time_read.append(otherStyleTime)
    gps['timestamp'] = timestamp_lst
    gps['readable_time'] = time_read
    gps = gps.sort_values(by='timestamp')  # add readable time info as a column, this one can be ignored when split by data info
    
    # split the dataframe by timestamp, get count, per 10 min
    time_slice = 60*10
    
    for row in range(gps.shape[0]):
        # belong to which time slice
        timestamp = gps.iloc[row][3]
        rounddown = int(timestamp/time_slice)
        start_time = rounddown*time_slice
        # count
        data = gps.iloc[row][2]
        lst = data.split(",")  #evey point
        for i in range(len(lst)):
            info = lst[i].strip().split(" ")
            pos_jing = eval(info[0])
            pos_wei = eval(info[1])
            direction = eval(info[3])
            timepoint = int(info[4])
            rd = int(timepoint/time_slice) * time_slice
            # calculate distance and math.fabs(direction-dict_road_pos[road_name[j]][2])<15
            # nearby = False
            for j in range(len(road_name)):
                if(cal_dist(pos_jing, pos_wei, dict_road_pos[road_name[j]][0], dict_road_pos[road_name[j]][1])<1e-3 and math.fabs(direction-dict_road_pos[road_name[j]][2])<40):
                    # nearby = True
                    try:
                        car_count.loc[rd][j] += 1
                    except:
                        t = [1 for i in range(len(road_name))]
                        t[j] = 2
                        dic = {}
                        for name,item in zip(road_name,t):
                            dic[name] = item
                        car_count.loc[rd] = dic
    try:
        car_count.to_csv(save_file, sep=',')
        print("save in round", round_count)
    except:
        print("failed to save")
    