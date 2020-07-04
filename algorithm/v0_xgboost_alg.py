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
from sklearn.decomposition import PCA
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
lzc = []
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

def train(train_X, train_y, eval_X, eval_y,road_index):
<<<<<<< HEAD
    dict_para = {'n_estim': [[50,50,120], [140,50,80], [60,140,80], [50,50,150], [50,140,140], [100,50,100],
                             [50,150,70], [120,50,150], [150,50,80], [50,70,80], [60,50,150], [50,50,80]],
                 'max_de': [[5,6,4], [6,4,6], [9,5,6], [3,4,3], [6,4,4], [2,3,7],
                            [5,6,4], [6,6,4], [5,6,4], [5,5,8], [5,6,4], [6,6,6]],
                 'min_child_wei': [[2,4,3], [1,4,1], [5,7,10], [1,9,7], [2,4,7], [2,9,9],
                                   [9,10,6], [3,8,8], [5,6,7], [6,7,8], [5,8,7], [3,3,3]],
                 'para_gamma': [[0.13,0.19,0.08], [0.0,0.04,0.0], [0.05,0.08,0.05], [0.04,0.01,0.05], [0.03,0.05,0.1], [0.09,0.04,0.06],
                                [0.01,0.03,0.05], [0.06,0.08,0.06], [0.01,0.02,0.0], [0.01,0.02,0.01], [0.03,0.1,0.0], [0.02,0.02,0.02]],
                 'subsample': [[0.6,1.0,1.0], [0.9,0.9,1.0], [0.8,0.8,0.9], [0.7,1.0,0.7], [0.6,1.0,0.9], [0.8,0.8,0.9],
                               [0.6,1.0,1.0], [0.6,0.7,0.8], [1.0,1.0,0.9], [0.7,0.8,1.0], [0.7,0.6,0.7], [0.8,0.8,0.8]],
                 'colsample_bytree': [[0.8,1.0,1.0], [1.0,1.0,1.0], [0.8,0.8,1.0], [0.8,0.9,1.0], [0.9,1.0,1.0], [0.4,1.0,0.8],
                                      [0.9,0.7,1.0], [1.0,1.0,1.0], [0.5,1.0,1.0], [0.8,1.0,1.0], [0.9,0.9,1.0], [0.8,1.0,0.7]],
                 'reg_lambda': [[10,30,50], [0,30,0], [10,30,50], [10,10,0], [10,0,30], [10,40,50],
                                [10,20,40], [50,50,50], [0,20,0], [20,60,50], [10,30,60], [30,60,30]],
                 'reg_alpha': [[0.7,0.0,0.6], [0.0,0.9,0.3], [0.7,0.2,0.4], [0.7,0.4,0.4], [0.7,0.3,0.0], [0.9,0.1,0.1],
                               [0.6,0.1,0.0], [0.2,0.7,1.0], [1.0,0.0,0.7], [0.5,0.6,0.0], [0.7,0.8,0.7], [0.0,0.0,0.0]],
                 'eta': [[0.2,0.3,0.1], [0.1,0.2,0.1], [0.1,0.1,0.1], [0.2,0.2,0.02], [0.1,0.07,0.1], [0.2,0.3,0.1],
                         [0.1,0.3,0.1], [0.2,0.2,0.07], [0.3,0.2,0.1], [0.2,0.2,0.2], [0.1,0.3,0.1], [0.1,0.1,0.1]]};
=======
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
    #model_total = []
>>>>>>> b00239e514f3841696b2e119fa5cca07fede15e0
    train_y = train_y.T
    eval_y = eval_y.T
    model = [0,0,0]
    #print(train_y.shape)
    for i in range(3):
        model[i] = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=160, min_child_weight=1,
                                    subsample=0.8, colsample_bytree=0.8, gamma=0,
                                    reg_alpha=0, reg_lambda=1)
        if i==0:
            model[i].fit(train_X, train_y[i],
                    eval_set = [(eval_X, eval_y[i])],
                    eval_metric='mae',
                    early_stopping_rounds = 10)
        elif i==1 or i==2:
            pre1 = model[i-1].predict(train_X)
            tmp = []
            for j in range(pre1.shape[0]):
                tmp.append([pre1[j]])
            tmp = np.array(tmp)
            train_X = np.hstack((train_X,tmp))
            
            pre1 = model[i-1].predict(eval_X)
            tmp = []
            for j in range(pre1.shape[0]):
                tmp.append([pre1[j]])
            tmp = np.array(tmp)
            eval_X = np.hstack((eval_X,tmp))
            
            
            model[i].fit(train_X, train_y[i],
                    eval_set = [(eval_X, eval_y[i])],
                    eval_metric='mae',
                    early_stopping_rounds = 10)
        
<<<<<<< HEAD
        model[i].fit(train_X, train_y[i],
                eval_set = [(eval_X, eval_y[i])],
                eval_metric='mae',
                early_stopping_rounds = 10)

=======
            
            
>>>>>>> b00239e514f3841696b2e119fa5cca07fede15e0
    models.append(model)
    #model_total.append(model)
    #models.append(model_total)
    return model
        
    

def gen_test(test_data):
    feature = []
    for row in range(0, test_data.shape[0]-5,6):
        tmp = []
        for i in range(6):
            tmp.append((test_data.iloc[row+i][0] % 86400) / 600) # related to time
            tmp.append(test_data.iloc[row+i][1])  # TTI
            tmp.append(test_data.iloc[row+i][2])  # num
            tmp.append(test_data.iloc[row+i][3])  # speed
        feature.append(tmp)
    feature = np.array(feature)
    return feature
    
    
    
    
'''   
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
'''

def evaluate(model, X, y):
    mae = 0
    y_t = y.T
    lst = predict(model,X)
    re = [[],[],[]]
    for i in range(0,len(lst),6):
        re[0].append(lst[i])
        re[1].append(lst[i+1])
        re[2].append(lst[i+2])
    for i in range(3):
        mae += mean_absolute_error(y_t[i],re[i])*len(re[i])
        
    print("mae:", mae/(X.shape[0]))
    return mae/(X.shape[0])

def predict(model,X_test):
    lst = []
    lst1 = []
    lst2 = []
    lst3 = []
    lst1 = model[0].predict(X_test)
    
    
    pre1 = lst1
    tmp = []
    for j in range(pre1.shape[0]):
        tmp.append([pre1[j]])
    tmp = np.array(tmp)
    X_test = np.hstack((X_test,tmp))
    lst2 = model[1].predict(X_test)
    
    pre1 = lst2
    tmp = []
    for j in range(pre1.shape[0]):
        tmp.append([pre1[j]])
    tmp = np.array(tmp)
    X_test = np.hstack((X_test,tmp))
    lst3 = model[2].predict(X_test)
    
    for i in range(lst1.shape[0]):
        lst.extend([lst1[i],lst2[i],lst3[i],0,0,0])
    return lst
    

        
        
        
            
                

def main():
    mae = 0
    #pre_df_lst = [0,0,0,0,0,0,0,0,0,0,0,0]
    pre_df_lst = [0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(12):
        train_data = pd.read_csv("../datasets/train_0103_"+road_name[i]+".csv", sep=',')
        train_data = train_data.sort_values(by = 'timestamp')
        test_data = pd.read_csv("../datasets/stage2_test_"+road_name[i]+".csv", sep=',')
        test_data = test_data.sort_values(by = 'timestamp')
        X, y = gen_train(train_data)#np array
        X_test = gen_test(test_data)
        y = y.T
        #PCA 
        #pca1 = PCA(n_components = 20)
        #pca2 = PCA(n_components = 20)
        #X = pca1.fit_transform(X)
        #X_test = pca2.fit_transform(X_test)
        
        
        #cluster num calculation
        #X_total = np.vstack((X,X_test))
       # N_CLUSTERS = 1
       # kmeans_model = KMeans(n_clusters = N_CLUSTERS)
       # kmeans_model.fit(X_total)
       # cluster_label = kmeans_model.labels_
        #print(cluster_label)
        
       # train_X, eval_X, train_y, eval_y,cluster_X,cluster_y = train_test_split(X,y,cluster_label[:X.shape[0]],test_size=0.25,random_state=1591545677)
        train_X, eval_X, train_y, eval_y = train_test_split(X, y, test_size=0.2, random_state=1591545677)
        #train_y.shape = (...,3)
        
        model = train(train_X, train_y, eval_X, eval_y,i)
        #print(X_test)
        pre_df = predict(model,X_test)
        #print(len(pre_df),test_data.shape[0])
        test_data['predict'] = pre_df
        df = pd.DataFrame()
        df['TTI'] = None
        for row in range(0,test_data.shape[0]-5,6):
            tss = test_data.iloc[row+5][0]
            df.loc[tss+600] = test_data.iloc[row][4]
            df.loc[tss+600*2] = test_data.iloc[row+1][4]
            df.loc[tss+600*3] = test_data.iloc[row+2][4]
        pre_df_lst[i] = df
        #pre_df_lst[i] = gen_test(model, test_data)
        #pd.DataFrame.to_csv(pre_df_lst[i], "D:/test_data/"+road_name[i]+"_pred.csv", sep=',')
    #print(pre_df_lst[0])
    #lzc = pre_df_lst[0]
    
    
    noLabel = pd.read_csv("../stage2_data/stage2/toPredict_noLabel_stage2.csv", sep=',')
    result = pd.DataFrame()
    result['TTI'] = None
    for row in range(noLabel.shape[0]):
        road_id = noLabel.loc[row, 'id_road']
        num = dict_road_index[road_id]
        time_str = noLabel.loc[row, 'time']
        timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(timeArray)
        #print(timestamp)
        try:
            x = pre_df_lst[num].loc[timestamp]['TTI']
            result.loc[row] = x
        except:
            assert(0)
    #print(time_cost)
    #print(result)
<<<<<<< HEAD
    result.to_csv("../model_result/xgb5.csv")
=======
    result.to_csv("../model_result/xgb_test.csv")
>>>>>>> b00239e514f3841696b2e119fa5cca07fede15e0
    #pd.DataFrame.to_csv(noLabel['pred'], "D:/test_data/pred_TTI3.csv", sep=',')

if __name__ == "__main__":
    main()