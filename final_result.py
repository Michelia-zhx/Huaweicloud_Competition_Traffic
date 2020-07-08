import pandas as pd
import numpy as np

rf = pd.read_csv("./model_result/rf.csv")
lg = pd.read_csv("./model_result/lgbm_test3_1872.csv")
#pca = pd.read_csv("./model_result/pca.csv")
boost = pd.read_csv("./model_result/xgb6.csv")
b2 = pd.read_csv("./model_result/xgb4_1803.csv")
b3 = pd.read_csv("./model_result/xgb9.csv")
re = pd.DataFrame()

tti = []
for i in range(rf.shape[0]):
    #x = 0.1*rf.iloc[i][1] + 0.45*lg.iloc[i][1] + 0.045*pca.iloc[i][1] + 0.405*boost.iloc[i][1]
    #x = 0.05*rf.iloc[i][1] + 0.95*lg.iloc[i][1]
    #x = 0.4*boost.iloc[i][1]+0.25*lg.iloc[i][1] + 0.05*rf.iloc[i][1]+0.3*b2.iloc[i][1]  0.18
    x = 0.25*lg.iloc[i][1] + 0.05*rf.iloc[i][1]+0.3*b2.iloc[i][1]+0.4*boost.iloc[i][1] #0.173
    
    tti.append(x)

re['id_sample'] = [i for i in range(rf.shape[0])]
re['TTI'] = tti
re.to_csv("./final10.csv",index = None)


    


