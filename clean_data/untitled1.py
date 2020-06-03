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
road_name = ['NanPing_W2E','NanPing_E2W','FuLong_S2N','FuLong_N2S','LiuXian_W2E','LiuXian_E2W','YuLong_S2N',
             'YuLong_N2S','XinQu_S2N','XinQu_N2S','ZhiYuan_S2N','ZhiYuan_N2S']
assert(len(road_name)==12)
dict_road_pos = {'NanPing_W2E':(114.015136,22.587219),'NanPing_E2W':(114.015136,22.587219),
                 'FuLong_S2N':(114.015666,22.603308),'FuLong_N2S':(114.015666,22.603308),
                 'LiuXian_W2E':(114.027811,22.612045),'LiuXian_E2W':(114.027811,22.612045),
                 'YuLong_S2N':(114.027826,22.605467),'YuLong_N2S':(114.027826,22.605467),
                 'XinQu_S2N':(114.03428,22.604796),'XinQu_N2S':(114.03428,22.604796),
                 'ZhiYuan_S2N':(114.025425,22.608589),'ZhiYuan_N2S':(114.025425,22.608589)
                 }

#--------------count from GPS--------
'''
number of cars passing the roads
columns: NanPing is a road name, W2E means from west to east
rows: per 30 min, index is start timestamp
'''

car_count = pd.DataFrame()
car_count['NanPing_W2E'] = None
car_count['NanPing_E2W'] = None
car_count['FuLong_S2N'] = None
car_count['FuLong_N2S'] = None
car_count['LiuXian_W2E'] = None
car_count['LiuXian_E2W'] = None
car_count['YuLong_S2N'] = None
car_count['YuLong_N2S'] = None
car_count['XinQu_S2N'] = None
car_count['XinQu_N2S'] = None
car_count['ZhiYuan_S2N'] = None
car_count['ZhiYuan_N2S'] = None

# how to insert a row: car_count.loc[0] = [1 for i in range(12)]
#car_count.loc[0] = [1 for i in range(12)]
#car_count.index = 15783792374
#-----------------function-------------
def cal_dist(pos_jing,pos_wei,road_x,road_y):
    return math.sqrt((pos_jing-road_x)**2 + (pos_wei-road_y)**2)


#---------------load the data-----------------
gps = pd.read_csv("D:/大一课程/机器学习/train_20191201_20191220/20191201_20191220.csv",nrows = 100);
gps.columns = ["user","order","info"]
#----------------wash the data-------------------
# add time info as a column
timestamp_lst = []
time_read = []
for row in range(gps.shape[0]):
    data = gps.iloc[row][2].strip("[").strip("]")
    gps.iloc[row][2] = data
    timestamp = data.split(",")[0].split(" ")[4]
    timestamp = int(timestamp)
    timeArray = time.localtime(timestamp)
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
    timestamp_lst.append(timestamp)
    time_read.append(otherStyleTime)
gps['timestamp'] = timestamp_lst
gps['readable_time'] = time_read
gps = gps.sort_values(by='timestamp')   #add readable time info as a column, this one can be ignored when split by data info

#split the dataframe by timestamp, get count, per 30min
time_slice = 60*30
for row in range(gps.shape[0]):
    #belong to which time slice
    timestamp = gps.iloc[row][3]
    rounddown = int(timestamp/time_slice)
    start_time = rounddown*time_slice
    #count
    data = gps.iloc[row][2]
    lst = data.split(",")  #evey point
    for i in range(len(lst)):
        info = lst[i].strip().split(" ")
        pos_jing = eval(info[0])
        pos_wei = eval(info[1])
        timepoint = int(info[4])
        rd = int(timepoint/time_slice)*time_slice
        #calculate distance
        temp_dist = [0 for i in range(12)]
        nearby = False
        for j in range(len(dict_road_pos)):
            if(cal_dist(pos_jing, pos_wei, dict_road_pos[road_name[j]][0], dict_road_pos[road_name[j]][1])<1e-4):
                nearby = True
                temp_dist[j] += 1
                try:
                    car_count.loc[rd][j] += 1
                except:
                    t = [0 for i in range(12)]
                    t[j] = 1
                    dic = {}
                    for name,item in zip(road_name,t):
                        dic[name] = item
                    car_count.loc[rd] = dic
                    
            #add this point to car_count
           # if(nearby==True):
           # array = [temp_dist]
           # df_temp = pd.DataFrame(array,columns = road_name,index = [rd])
              #  try:
                 #   car_count.loc[rd][j] += 1;
                        
    
    
