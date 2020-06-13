# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:20:19 2020

@author: zhanghanxiao
"""
#---------------------------dependencies-----------------------------
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from sklearn.cluster import KMeans
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

models = []

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

def train(train_X, train_y, eval_X, eval_y, cluster_X,cluster_eval,road_index):
    # params = {
    #     'booster': 'gbtree',
    #     'objective': 'reg:gamma',
    #     'gamma': 0.1,
    #     'max_depth': 6,
    #     'lambda': 3,
    #     'subsample': 0.7,
    #     'colsample_bytree': 0.7,
    #     'min_child_weight': 3,
    #     'eta': 0.1,
    #     'seed': 1000,
    #     'nthread': 4,
    # }
    # num_round = 1
    # # plst = params.items()
    # evallist = [(deval, 'eval'), (dtrain, 'train')]
    # bst = xgb.train(params, dtrain, num_round, evallist)
    # model_name = road_name[road_index] + ".model"
    # bst.save_model(model_name)
    # return bst
    model_total = []
    #train_y = train_y.T
    #eval_y = eval_y.T
    #print(train_y.shape)
    for clu in range(1):
        train_x_temp = []
        eval_x_temp = []
        train_y_temp = []
        eval_y_temp = []
        for i in range(cluster_X.shape[0]):
            if cluster_X[i]==clu:
                train_x_temp.append(train_X[i])
                train_y_temp.append(train_y[i])
            else:
                assert(0)
                
        for i in range(cluster_eval.shape[0]):
            if cluster_eval[i]==clu:
                eval_x_temp.append(eval_X[i])
                eval_y_temp.append(eval_y[i])
        train_y_temp = np.array(train_y_temp)
        eval_y_temp = np.array(eval_y_temp)
        train_y_temp =train_y_temp.T
        eval_y_temp = eval_y_temp.T
        model = [0, 0, 0]
        for i in range(3):
            model[i] = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=160, min_child_weight=1,
                                    subsample=0.8, colsample_bytree=0.8, gamma=0,
                                    reg_alpha=0, reg_lambda=1)
            train_x_temp =np.array(train_x_temp)
            #train_y_temp =np.array(train_y_temp[i])
            eval_x_temp = np.array(eval_x_temp)
            #eval_y_temp = np.array(eval_y_temp[i])
            
            model[i].fit(train_x_temp, train_y_temp[i],
                    eval_set = [(eval_x_temp, eval_y_temp[i])],
                    eval_metric='mae',
                    early_stopping_rounds = 10)
        model_total.append(model)
    models.append(model_total)
    return model_total

def gen_test(model, pred_df):
    pred_df['pred1'] = None
    pred_df['pred2'] = None
    pred_df['pred3'] = None
    for row in range(0, pred_df.shape[0], 6):
        feature = []
        tmp = []
        for i in range(0,6):
            tmp.append((pred_df.iloc[row+i][0] % 86400) / 600) # related to time
            tmp.append(pred_df.iloc[row+i][1])  # TTI
            tmp.append(pred_df.iloc[row+i][2])  # num
            tmp.append(pred_df.iloc[row+i][3])  # speed
        feature.append(tmp)
        x = np.array(feature)
        # print(x)
        for i in range(3):    
            ypred = model[i].predict(x)
            column = 'pred' + str(i+1)
            pred_df.loc[row,column] = ypred
    # print(pred_df)
    return pred_df

def evaluate(model, X, y,cluster):
    mae = 0
    for clu in range(1):
        X_t = []
        y_t = [[],[],[]]
        for i in range(cluster.shape[0]):
            if cluster[i]==clu:
                X_t.append(X[i])
                y_t[0].append(y[i][0])
                y_t[1].append(y[i][1])
                y_t[2].append(y[i][2])
        X_t = np.array(X_t)
        for i in range(3):
            y_pred = model[clu][i].predict(X_t)
            mae += mean_absolute_error(y_t[i], y_pred)*y_pred.shape[0]
    
    print("mae:", mae/X.shape[0])
    return mae/X.shape[0]

def main():
    mae = 0
    pre_df_lst = [0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(12):
        train_data = pd.read_csv("D:/test_data/train_data/train_"+road_name[i]+".csv", sep=',')
        train_data = train_data.sort_values(by = 'timestamp')
        test_data = pd.read_csv("D:/test_data/train_data/test_"+road_name[i]+".csv", sep=',')
        test_data = test_data.sort_values(by = 'timestamp')
        X, y = gen_train(train_data)#np array
        y = y.T
        #cluster num calculation
        kmeans_model = KMeans(n_clusters = 1)
        kmeans_model.fit(X)
        cluster_label = kmeans_model.labels_
        #print(cluster_label)
        
        train_X, eval_X, train_y, eval_y,cluster_X,cluster_y = train_test_split(X,y,cluster_label,test_size=0.25, random_state=0)
        
        
        model = train(train_X, train_y, eval_X, eval_y, cluster_X,cluster_y,i)
        mae += evaluate(model, eval_X, eval_y,cluster_y)
        #pre_df_lst[i] = gen_test(model, test_data)
        #pd.DataFrame.to_csv(pre_df_lst[i], "D:/test_data/"+road_name[i]+"_pred.csv", sep=',')
    print(mae / 12)
    
'''
    noLabel = pd.read_csv("D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/traffic/toPredict_noLabel.csv", sep=',')
    noLabel['pred'] = None
    for row in range(noLabel.shape[0]):
        road_id = noLabel.loc[row, 'id_road']
        time_str = noLabel.loc[row, 'time']
        timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(timeArray)
        column = 'pred' + str(row % 3 + 1)
        choose_row = timestamp - (3600 + (row % 3) * 600)
        try:
            pre_TTI = pre_df_lst[dict_road_index[road_id]].set_index('timestamp')
            pre_TTI = pre_TTI.loc[choose_row, column]
            # print(pre_TTI)
            noLabel.loc[row, 'pred'] = pre_TTI
        except:
            continue
    pd.DataFrame.to_csv(noLabel['pred'], "D:/test_data/pred_TTI3.csv", sep=',')
'''
if __name__ == "__main__":
    main()