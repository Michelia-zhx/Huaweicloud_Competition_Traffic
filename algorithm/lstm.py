# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:31:03 2020

@author: 98061
"""

import pandas as pd
import numpy as np
import time 
import datetime
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
#----------------LSTM--------------
class LstmManager(torch.nn.Module):
    def __init__(self,input_dim,hidden_size,output_size):
        super(LstmManager, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim,hidden_size)
        self.linearMap = torch.nn.Linear(hidden_size,output_size)
        
    def forward(self,info):#info.dtype=float32, 要用torch.tensor(...,dtype=torch.float32)
        lstm_out,_ = self.lstm(info.view(len(info),1,-1))
        linear_out = self.linearMap(lstm_out.view(len(lstm_out),1,-1))
        return linear_out
    
    


#----------------const value---------------
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
model_lst = []
#----------------load train data--------------
for name in road_name:
    path = "D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/datasets/train_"+name+".csv"
    train = pd.read_csv(path)
    feature = []
    label = []
    for row in range(6,train.shape[0]-2):
        lst = []
        for i in range(1,7):
            lst.append(train.iloc[row-i][3]) #speed
            lst.append(train.iloc[row-i][2])  #num
            lst.append(train.iloc[row-i][1])#TTI
        label.append([train.iloc[row][1],train.iloc[row+1][1],train.iloc[row+2][1]])
        feature.append(lst)   
#----------------train-------------------
    model = LstmManager(18,30,3)
    loss_function = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_value = 0
    for epoch in range(20):
        for batch in range(int(len(feature)/100)):
            model.zero_grad()
            tmp_f = []
            tmp_l = []
            for i in range(100):
                tmp_f.append([feature[i]])
                tmp_l.append([label[i]])
            info = torch.tensor(tmp_f,dtype = torch.float32)
            target = torch.tensor(tmp_l,dtype = torch.float32)
            out = model(info)
            loss = loss_function(out,target)
           # if batch%10==0:
               # print(loss)
            #loss_value = loss.tolist()
            loss.backward()
            optimizer.step()
        print('finish epoch',epoch,loss_value,sep = ' ')
    model_lst.append(model)

#------------------load test data---------------
# run get 0.4
noLabel = pd.read_csv('D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/traffic/toPredict_noLabel.csv')

def predict(road_id,timestamp,train_TTI,test_gps,lst):
    n = dict_id2num[road_id]
    rf = lst[n]
    feature = []
    lst_t = []
    for i in range(1,7):
        timeslice = 60*i*10
        ts = timestamp-timeslice
        tmp = train_TTI[(train_TTI.timestamp==ts) & (train_TTI.id_road==road_id)]
        speed = tmp['speed'].tolist()[0]
        TTI = tmp['TTI'].tolist()[0]
        road_name = dic[road_id]
        car_num = test_gps.loc[ts][road_name]
        lst_t.append(speed)
        lst_t.append(car_num)
        lst_t.append(TTI)
        #print('b')
   # print('a')
    feature.append([lst_t])
    info = torch.tensor(feature,dtype=torch.float32)
    re = lst[n](info)
    
    return re

test_gps = pd.read_csv('D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/processed_test_data/pro_toPredict_gps.csv')
extend_name = ['times']
extend_name.extend(road_name)
test_gps.columns = extend_name
test_gps.index = test_gps.times
test_gps = test_gps.drop('times',axis = 1)
test_gps = test_gps.sort_index()
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
    x = predict(road_id,timestamp,train_TTI,test_gps,model_lst)
    x = x.tolist()
    label_predict.append(x[0][0][0])
    label_predict.append(x[0][0][1])
    label_predict.append(x[0][0][2])
    
      
noLabel['TTI'] = label_predict
noLabel = noLabel.drop('time',axis = 1)
noLabel = noLabel.drop('id_road',axis = 1)
noLabel.to_csv("D:/test_data/TTI6.csv",index = None)