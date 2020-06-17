# -*- coding: utf-8 -*-
"""
Created on Thu Jun  11 17:10:19 2020

@author: zhanghanxiao
"""
#-----------------------dependencies---------------------------
import pandas as pd

#------------------------const value --------------------------
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

#-----------------------join training sets---------------------
for i in range(len(road_name)):
    roadname = road_name[i]

    filename = '../datasets/train_0103_' + roadname + '.csv'
    train_0103 = pd.read_csv(filename, sep=',')
    len_0103 = int(train_0103.shape[0] / 6) * 6 - 1
    train_0103 = train_0103[:len_0103]

    filename = '../datasets/train_12_' + roadname + '.csv'
    train_12 = pd.read_csv(filename, sep=',')
    len_12 = int(train_12.shape[0] / 6) * 6 - 1
    train_12 = train_12[:len_12]

    trainset = pd.concat([train_0103, train_12], axis=0).reset_index().drop(['index'], axis=1)
    save_name = '../datasets/train_' + roadname + '.csv'
    pd.DataFrame.to_csv(trainset, save_name, sep=',')