# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 08:57:52 2020

@author: 98061
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import lightgbm as lgb
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
dic = {276183:'NanPing_W2E',276184:'NanPing_E2W',
                275911:'FuLong_S2N',275912:'FuLong_N2S',
                276240:'LiuXian_W2E',276241:'LiuXian_E2W',
                276264:'YuLong_S2N',276265:'YuLong_N2S',
                276268:'XinQu_S2N',276269:'XinQu_N2S',
                276737:'ZhiYuan_S2N',276738:'ZhiYuan_N2S'}
#--------------------model---------------------------------------
models = []
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression_l1',  # 目标函数
    'metric': {'l1'},  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.1,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
#----------------- function ----------------------------------
def train(train_df,test_df,params):
    lst_f = []
    lst1 = []
    lst2 = []
    lst3 = []
    for row in range(6,train_df.shape[0]-2,3):
        feature= []
        for i in range(1,7):
            feature.append(train_df.iloc[row-i][3])#speed
            feature.append(train_df.iloc[row-i][2])#car
            feature.append(train_df.iloc[row-i][1])#TTI
            timestamp = train_df.iloc[row-i][0]
            ts = timestamp % (24*60*60)
            feature.append(ts)
        lst1.append(train_df.iloc[row][1])
        lst2.append(train_df.iloc[row+1][1])
        lst3.append(train_df.iloc[row+2][1])
        lst_f.append(feature)
    lgb_train1 = lgb.Dataset(lst_f,lst1)
    lgb_train2 = lgb.Dataset(lst_f,lst2)
    lgb_train3 = lgb.Dataset(lst_f,lst3)
    
    lst_f = []
    lst1 = []
    lst2 = []
    lst3 = []
    for row in range(6,test_df.shape[0]-2,3):
        feature= []
        for i in range(1,7):
            feature.append(test_df.iloc[row-i][3])#speed
            feature.append(test_df.iloc[row-i][2])#car
            feature.append(test_df.iloc[row-i][1])#TTI
            timestamp = test_df.iloc[row-i][0]
            ts = timestamp % (24*60*60)
            feature.append(ts)
        lst1.append(test_df.iloc[row][1])
        lst2.append(test_df.iloc[row+1][1])
        lst3.append(test_df.iloc[row+2][1])
        lst_f.append(feature)
    lgb_test1 = lgb.Dataset(lst_f,lst1)
    lgb_test2 = lgb.Dataset(lst_f,lst2)
    lgb_test3 = lgb.Dataset(lst_f,lst3)
    
    gbm1 = lgb.train(params, lgb_train1, num_boost_round=30, valid_sets=lgb_test1, early_stopping_rounds=5)
    print("---------------------")
    gbm2 = lgb.train(params, lgb_train2, num_boost_round=30, valid_sets=lgb_test2, early_stopping_rounds=5)
    print("---------------------")
    gbm3 = lgb.train(params, lgb_train3, num_boost_round=30, valid_sets=lgb_test3, early_stopping_rounds=5)
    models.append([gbm1,gbm2,gbm3])
    
def predict(road_id,timestamp,train_TTI,test_gps,models):
    n = dict_road_index[road_id]
    gbm1 = models[n][0]
    gbm2 = models[n][1]
    gbm3 =models[n][2]
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
        y = ts%(86400)
        lst_t.append(y)
        #print('b')
   # print('a')
    feature.append(lst_t)
    test_x = np.array(feature)
    y_pred1 = gbm1.predict(test_x, num_iteration=gbm1.best_iteration)
    y_pred2 = gbm2.predict(test_x, num_iteration=gbm2.best_iteration)
    y_pred3= gbm3.predict(test_x, num_iteration=gbm3.best_iteration)
    #print(y_pred1)
    return (y_pred1,y_pred2,y_pred3)

def main():
    for name in range(len(road_name)):
        print(road_name[name])
        path = "../datasets/train_0103_"+road_name[name]+".csv"
        train_df = pd.read_csv(path)
        test_df = pd.read_csv(path)
        train_df = train_df.sort_values(by ='timestamp')
        test_df = test_df.sort_values(by='timestamp')
        train(train_df,test_df,params)
        
    noLabel = pd.read_csv("../stage2_data/stage2/toPredict_noLabel_stage2.csv")
    train_TTI = pd.read_csv('../stage2_data/stage2/toPredict_train_TTI_stage2.csv')
    timestamps = []
    for row in range(train_TTI.shape[0]):
        tss1 = train_TTI.iloc[row][3]
        timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
        timestamp = int(time.mktime(timeArray))
        timestamps.append(timestamp)
    train_TTI['timestamp'] = timestamps
    test_gps = pd.read_csv('../processed_test_data/stage2_pro_toPredict_gps.csv')
    extend_name = ['times']
    extend_name.extend(road_name)
    test_gps.columns = extend_name
    test_gps.index = test_gps.times
    test_gps = test_gps.drop('times',axis = 1)
    test_gps = test_gps.sort_index()
    
    predict_label = []
    for row in range(0,noLabel.shape[0]-2,3):
        tss1 = noLabel.iloc[row][2]
        timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
        timestamp = int(time.mktime(timeArray))
        road_id = noLabel.iloc[row][1]
        x = predict(road_id,timestamp,train_TTI,test_gps,models)
        predict_label.append(x[0][0])
        predict_label.append(x[1][0])
        predict_label.append(x[2][0])
    noLabel['TTI'] = predict_label
    noLabel = noLabel.drop('time',axis = 1)
    noLabel = noLabel.drop('id_road',axis = 1)
    noLabel.to_csv("../model_result/lgbm.csv",index = None)
        
       
main()
'''
tti5 = pd.read_csv('D:/test_data/TTI5.csv')
tti3 =pd.read_csv('D:/test_data/pred_TTI2.csv')
error = 0
for row in range(tti5.shape[0]):
    error += math.fabs(tti5.iloc[row][1]-tti3.iloc[row][1])
print(error/tti5.shape[0])
'''