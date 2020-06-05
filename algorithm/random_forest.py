# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:40:26 2020

@author: 98061
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
import datetime

road_name = ['NanPing_W2E','NanPing_E2W','FuLong_S2N','FuLong_N2S','LiuXian_W2E','LiuXian_E2W','YuLong_S2N',
             'YuLong_N2S','XinQu_S2N','XinQu_N2S','ZhiYuan_S2N','ZhiYuan_N2S']

dict_id2num = {276183:0,276184:1,
                275911:2,275912:3,
                276240:4,276241:5,
                276264:6,276265:7,
                276268:8,276269:9,
                276737:10,276738:11}
dic = {276183:'NanPing_W2E',276184:'NanPing_E2W',
                275911:'FuLong_S2N',275912:'FuLong_N2S',
                276240:'LiuXian_W2E',276241:'LiuXian_E2W',
                276264:'YuLong_S2N',276265:'YuLong_N2S',
                276268:'XinQu_S2N',276269:'XinQu_N2S',
                276737:'ZhiYuan_S2N',276738:'ZhiYuan_N2S'}

rf1 = RandomForestRegressor()
rf2= RandomForestRegressor()
rf3 = RandomForestRegressor()
rf4 = RandomForestRegressor()
rf5 = RandomForestRegressor()
rf6 = RandomForestRegressor()
rf7 = RandomForestRegressor()
rf8 = RandomForestRegressor()
rf9 = RandomForestRegressor()
rf10 = RandomForestRegressor()
rf11 = RandomForestRegressor()
rf12 = RandomForestRegressor()
lst = []
lst.append(rf1)
lst.append(rf2)
lst.append(rf3)
lst.append(rf4)
lst.append(rf5)
lst.append(rf6)
lst.append(rf7)
lst.append(rf8)
lst.append(rf9)
lst.append(rf10)
lst.append(rf11)
lst.append(rf12)

#--------------------train---------------
def train(df,rf1):
    feature = []
    label = []
    for row in range(6,df.shape[0]-2,3):
        lst = []
        for i in range(1,7):
            lst.append(df.iloc[row-i][3]) #speed
            lst.append(df.iloc[row-i][2])  #num
            lst.append(df.iloc[row-i][1])#TTI
        label.append([df.iloc[row][1],df.iloc[row+1][1],df.iloc[row+2][1]])
        feature.append(lst)
    rf1.fit(feature,label)

def predict(road_id,timestamp,train_TTI,test_gps,lst):
    n = dict_id2num[road_id]
    rf = lst[n]
    feature = []
    lst_t = []
    for i in range(1,7):
        timeslice = 60*i*10
        ts = timestamp-timeslice
        tmp = train_TTI[(train_TTI.timestamp==ts) & (train_TTI.id_road==road_id)]
        speed = tmp['speed']
        TTI = tmp['TTI']
        road_name = dic[road_id]
        car_num = test_gps.loc[ts][road_name]
        lst_t.append(speed)
        lst_t.append(car_num)
        lst_t.append(TTI)
        #print('b')
   # print('a')
    feature.append(lst_t)
    re = rf.predict(feature)
    
    return re
        
        
        
    

for i in range(len(road_name)):
    path = "D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/datasets/dataset_"+road_name[i]+".csv"
    df = pd.read_csv(path)
    df = df.sort_values(by = 'timestamp')
    train(df,lst[i])
  
test_gps = pd.read_csv('D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/processed_test_data/pro_toPredict_gps.csv')
extend_name = ['times']
extend_name.extend(road_name)
test_gps.columns = extend_name
test_gps.index = test_gps.times
test_gps = test_gps.drop('times',axis = 1)
test_gps = test_gps.sort_index()


noLabel = pd.read_csv('D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/traffic/toPredict_noLabel.csv')
train_TTI = pd.read_csv('D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/traffic/toPredict_train_TTI.csv')
timestamps = []
for row in range(train_TTI.shape[0]):
    tss1 = train_TTI.iloc[row][3]
    timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(timeArray))
    timestamps.append(timestamp)
train_TTI['timestamp'] = timestamps

label_predict = []
for row in range(0,noLabel.shape[0]-2,3):
    tss1 = noLabel.iloc[row][2]
    timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(timeArray))
    road_id = noLabel.iloc[row][1]
    x = predict(road_id,timestamp,train_TTI,test_gps,lst)
    label_predict.append(x[0][0])
    label_predict.append(x[0][1])
    label_predict.append(x[0][2])
noLabel['TTI'] = label_predict
noLabel = noLabel.drop('time',axis = 1)
noLabel = noLabel.drop('id_road',axis = 1)
noLabel.to_csv("D:/test_data/TTI3.csv",index = None)






    