# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 08:55:06 2020

@author: 98061
"""

start=    1546272000
end= 1548950400

import pandas as pd
import numpy as np


road_name = ['NanPing_W2E', 'NanPing_E2W', 'FuLong_S2N', 'FuLong_N2S', 'LiuXian_W2E', 'LiuXian_E2W',
             'YuLong_S2N', 'YuLong_N2S', 'XinQu_S2N', 'XinQu_N2S', 'ZhiYuan_S2N', 'ZhiYuan_N2S']
assert(len(road_name)==12)

for i in range(len(road_name)):
    print(road_name[i])
    name = '../datasets/train_0103_'+road_name[i]+'.csv'
    df = pd.read_csv(name)
    lst = df.timestamp
    df.index = lst
    for j in range(len(lst)):
        ts = lst[j]
        if ts<start or ts>end:
            df = df.drop(ts)
    df = df.drop('timestamp',axis = 1)
    df.to_csv('../datasets/train_01_'+road_name[i]+'.csv')
    print(road_name[i],'finish',sep = ' ')
    
    
