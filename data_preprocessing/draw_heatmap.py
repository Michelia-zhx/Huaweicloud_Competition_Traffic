# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 02:40:39 2020

@author: zhanghanxiao
"""

#---------------dependencies----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import math
import folium
import webbrowser
from folium.plugins import HeatMap

#---------------collect data----------
def collect_data():
    time_slice_data = [[], [], [], [], [], []] # 6:00-7:00, 9:00-10:00, 12:00-13:00, 15:00-16:00, 18:00-19:00, 21:00-22:00每次只画一个时间
    
    filename="../../20191201_20191220.csv"
    raw_data = pd.read_csv(filename, iterator = True)
    chunk_size  = 1000
    round_count = 0
    while(True):
        round_count += 1
        try:
            gps = raw_data.get_chunk(chunk_size)
            print(round_count)
        except:
            break
        gps.columns = ["order","user","info"]
    
        # ----------------clean the data-------------------
        # add time info as a column
        for row in range(gps.shape[0]):
            data = gps.iloc[row][2].strip("[").strip("]").split(",")
            for i in range(1, len(data)):
                timestamp = int(data[i].split(" ")[5])
                day_min = timestamp % 86400
                pos_jing = eval(data[i].split(" ")[1])
                pos_wei = eval(data[i].split(" ")[2])
                speed = eval(data[i].split(" ")[3])
                itm = [pos_jing, pos_wei, 100-speed]
                if day_min >= 21600 and day_min < 25200:
                    time_slice_data[0].append(itm)
                elif day_min >= 32400 and day_min < 36000:
                    time_slice_data[1].append(itm)
                elif day_min >= 43200 and day_min < 46800:
                    time_slice_data[2].append(itm)
                elif day_min >= 54000 and day_min < 57600:
                    time_slice_data[3].append(itm)
                elif day_min >= 64800 and day_min < 68400:
                    time_slice_data[4].append(itm)
                elif day_min >= 75600 and day_min < 79200:
                    time_slice_data[5].append(itm)
                else:
                    continue
        try:
            for i in range(len(time_slice_data)):
                df = pd.DataFrame(time_slice_data[i])
                filename = '../processed_train_data/heatmap_data/df' + str(i) + '.csv'
                pd.DataFrame.to_csv(df, filename, sep=',')
            print("save!")
        except:
            continue

def draw_heatmap():
    for i in range(0, 6):
        filename = '../processed_train_data/heatmap_data/df' + str(i) + '.csv'
        pos_data = pd.read_csv(filename, iterator=True)
        chunk_size = 100000
        data = pos_data.get_chunk(chunk_size)
        data.columns = ['index', 'pos_jing', 'pos_wei', '100-speed']
        data = data.drop(['index'], axis=1)
        jing = np.array(data['pos_jing'][0:len(data)], dtype=float)
        wei = np.array(data['pos_wei'][0:len(data)], dtype=float)
        neg_speed = np.array(data['100-speed'][0:len(data)], dtype=float)
        to_draw = [[wei[j]+0.0029, jing[j]-0.005, (neg_speed[j]-90)*10] for j in range(len(data))]
        print(to_draw)
        map_osm = folium.Map(location=[22.60185, 114.0124], zoom_start=14.4)
        HeatMap(to_draw, radius=10).add_to(map_osm) 
        file_path = '../processed_train_data/heatmap_data/heatmap' + str(i) + '.html'
        map_osm.save(file_path)   #保存本地

def main():
    # collect_data()
    draw_heatmap()

if __name__ == "__main__":
    main()