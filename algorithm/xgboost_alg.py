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
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import time
from copy import deepcopy

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

pre_df_lst = []
models = []

def gen_train(train_data):
    feature = []
    label = [[], [], []]
    for row in range(6,train_data.shape[0]-2, 3):
        tmp = []
        for i in range(1,7):
            # tmp.append((train_data.iloc[row-i][0] % 86400) % 600)  # time
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

def train(train_X, train_y, road_index):
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
    model = [0, 0, 0]
    for i in range(3):
        model[i] = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, objective='reg:gamma')
        model[i].fit(train_X, train_y[i])
        model.append(model[i])
    return model

def gen_test(model, pred_df):
    pred_df['pred1'] = None
    pred_df['pred2'] = None
    pred_df['pred3'] = None
    for row in range(0, pred_df.shape[0], 6):
        feature = []
        tmp = []
        for i in range(0,6):
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

def main():
    for i in range(len(road_name)):
        train_data = pd.read_csv("../datasets/train_"+road_name[i]+".csv", sep=',')
        train_data = train_data.sort_values(by = 'timestamp')
        test_data = pd.read_csv("../datasets/test_"+road_name[i]+".csv", sep=',')
        test_data = test_data.sort_values(by = 'timestamp')
        X, y = gen_train(train_data)
        # train_size = int(X.shape[0] * 0.8)
        # train_X = X[0:train_size]
        # train_y = y[0:train_size]
        # eval_X  = X[train_size:]
        # eval_y  = y[train_size:]
        # dtrain  = xgb.DMatrix(train_X, label=train_y)
        # deval   = xgb.DMatrix(eval_X, label=eval_y)
        model = train(X, y, i)
        pre_df = gen_test(model, test_data)
        pre_df_lst.append(pre_df)
        pd.DataFrame.to_csv(pre_df, road_name[i]+"_pred.csv", sep=',')
    noLabel = pd.read_csv("../traffic/toPredict_noLabel.csv", sep=',')
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
    pd.DataFrame.to_csv(noLabel['pred'], "pred_TTI.csv", sep=',')

if __name__ == "__main__":
    main()