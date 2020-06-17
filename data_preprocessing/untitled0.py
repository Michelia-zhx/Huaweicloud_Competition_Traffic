# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 02:40:39 2020

@author: zhanghanxiao
"""

#---------------dependencies----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import math

#---------------collect data----------
time_slice_data = [] # 6:00-7:00, 9:00-10:00, 12:00-13:00, 15:00-16:00, 


filename="../../20191201_20191220.csv"
raw_data = pd.read_csv(filename, iterator = True)
chunk_size  = 1000
round_count = 0
while(True):
    round_count += 1
    try:
        gps = raw_data.get_chunk(chunk_size)
        print(round_count)
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
                #if(cal_dist(pos_jing, pos_wei, dict_road_pos[road_name[j]][0], dict_road_pos[road_name[j]][1])<1e-3 and math.fabs(direction-dict_road_pos[road_name[j]][2])<40):
                data = dict_road_pos[road_name[j]]
                if(inArea(data[1],data[2],data[3],data[4],(pos_jing,pos_wei)) and math.fabs(direction-data[0])<40):
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