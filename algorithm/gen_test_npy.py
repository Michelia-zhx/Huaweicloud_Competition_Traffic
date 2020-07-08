# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:18:43 2020

@author: 98061
"""

#-------------------------dependencies--------------------------
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime

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

related_roads = {0:(3,5,6),1:(9,11,2),
                 2:(1,7,4),3:(5,8,0),
                 4:(2,1,9),5:(8,6,3),
                 6:(0,3,10),7:(11,9,1),
                 8:(6,0,5),9:(4,11,7),
                 10:(6,0,5),11:(4,2,7)}


def get_day(date):
    ltime = time.localtime(int(date))    
    dateymd = time.strftime("%Y-%m-%d", ltime)    
    #print(dateymd)    
    #print(datetime.strptime(dateymd, "%Y-%m-%d").weekday())
    return datetime.strptime(dateymd, "%Y-%m-%d").weekday()


#--------------------generate train array-----------------------
def gen_test(train_data,road_index):
    in1 = related_roads[road_index][0]
    in2 = related_roads[road_index][1]
    out = related_roads[road_index][2]
    
    df_in1 = pd.read_csv("../datasets/stage2_test_"+road_name[in1]+".csv")
    df_in2 = pd.read_csv("../datasets/stage2_test_"+road_name[in2]+".csv")
    df_out = pd.read_csv("../datasets/stage2_test_"+road_name[out]+".csv")
    
    df_in1 = df_in1.sort_values(by='timestamp')
    df_in1.index = df_in1.timestamp
    df_in1 = df_in1.drop('timestamp',axis = 1)
    
    df_in2 = df_in2.sort_values(by='timestamp')
    df_in2.index = df_in2.timestamp
    df_in2 = df_in2.drop('timestamp',axis = 1)
    
    df_out = df_out.sort_values(by='timestamp')
    df_out.index = df_out.timestamp
    df_out = df_out.drop('timestamp',axis = 1)
    
    
    feature = []
    print(train_data.shape[0])
    for row in range(6, train_data.shape[0]+1, 6):
        tmp = []
        rel = [[],[],[]]  # features of related three roads
        yesterday = []  # features of yesterday
        # last_week = []  # features of last week
        time_start = train_data.iloc[row-6][0]
        try:  
            for add in range(0,3600,600):
                ts = time_start + add
                rel[0].extend([df_in1.loc[ts]['TTI'],df_in1.loc[ts]['car_count'],df_in1.loc[ts]['speed']])
            for add in range(0,3600,600):
                ts = time_start + add
                rel[1].extend([df_in2.loc[ts]['TTI'],df_in2.loc[ts]['car_count'],df_in2.loc[ts]['speed']])
            for add in range(0,3600,600):
                ts = time_start + add
                rel[2].extend([df_out.loc[ts]['TTI'],df_out.loc[ts]['car_count'],df_out.loc[ts]['speed']])
        except:
            print("fail to try1:", row)
            continue
        
        #get first 18 dim
        for i in range(6,0,-1):
            #tmp.append((train_data.iloc[row-i][0] % 86400) / 600) # related to time
            ts = train_data.iloc[row-i][0]
            tmp.append(train_data.iloc[row-i][1])  # TTI
            tmp.append(train_data.iloc[row-i][2])  # num
            tmp.append(train_data.iloc[row-i][3])  # speed
        
        #get next 54 dim
        for i in range(3):
            tmp.extend(rel[i])
        #get yesterday 18dim
        try:
            time_yesterday = time_start - 24*60*60
            for add in range(0,3600,600):
                ts = time_yesterday + add
                yesterday.extend([train_data.loc[ts]['TTI'],train_data.loc[ts]['car_count'],train_data.loc[ts]['speed']])
            tmp.extend(yesterday)
        except:
            for i in range(6,0,-1):
                tmp.append(train_data.iloc[row-i][1])  # TTI
                tmp.append(train_data.iloc[row-i][2])  # num
                tmp.append(train_data.iloc[row-i][3])  # speed
        #get next 18dim
        # try:
        #     time_lastweek = time_start - 24*60*60*7
        #     for add in range(0,3600,600):
        #         ts = time_lastweek + add
        #         last_week.extend([train_data.loc[ts]['TTI'],train_data.loc[ts]['car_count'],train_data.loc[ts]['speed']])
        #     tmp.extend(last_week)
        # except:
        #     print("fail to try3:", row)
        #     continue
        #get last 2 dim
        tmp.append(get_day(time_start))
        tmp.append((time_start % 86400) / 600)

        feature.append(tmp)
        
    feature = np.array(feature)
    
    return feature

def main():
    for i in range(0,len(road_name)):
        print("start to process",road_name[i],sep = ' ')
        train_data = pd.read_csv("../datasets/stage2_test_"+road_name[i]+".csv", sep=',')
        train_data = train_data.sort_values(by = 'timestamp')
        train_data.index = train_data.timestamp
        X = gen_test(train_data,i)
        X_filename = 'train_array/test_X_0103_' + road_name[i] + '.npy'
        np.save(X_filename, X)
        print("save", i)

if __name__ == "__main__":
    main()