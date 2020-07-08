# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:47:09 2020

@author: 98061
"""
import pandas as pd
import numpy as np
import math
tti5 = pd.read_csv('./model_result/lgbm0708.csv')
tti3 =pd.read_csv('./model_result/lgbm_test3_1872.csv')
#noLabel  = pd.read_csv("D:/HW_AI_traffic/Huaweicloud_Competition_Traffic/traffic/toPredict_noLabel.csv")
    
count = 0
error = 0
for row in range(tti5.shape[0]):
   # if noLabel.iloc[row][1]!=276737 and noLabel.iloc[row][1]!=276738:
        #count += 1
    error += math.fabs(tti5.iloc[row][1]-tti3.iloc[row][1])
print(error/tti5.shape[0])