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
    
    


#----------------load train data---------------
train = pd.read_csv("D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/datasets/train_NanPing_W2E.csv")
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
for epoch in range(10):
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
        loss.backward()
        optimizer.step()
    print('finish epoch',epoch,sep = ' ')

#------------------load test data---------------
# not run yet
noLabel = pd.read_csv('D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/traffic/toPredict_noLabel.csv')
noLabel = noLabel[(noLabel.id_road==276183)]

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
    feature.append([lst_t])
    re = lst[n](feature)
    
    return re




label_predict = []
for row in range(0,noLabel.shape[0]-2,3):
    tss1 = noLabel.iloc[row][2]
    timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(timeArray))
    
    x = predict(road_id,timestamp,train_TTI,test_gps,lst)
    x = x.tolist()
    label_predict.append(x[0][0][0])
    label_predict.append(x[0][0][1])
    label_predict.append(x[0][0][2])
    
        
    