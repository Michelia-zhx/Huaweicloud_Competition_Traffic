# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:19:04 2020

@author: zhanghanxiao
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(1)
row_num = 256
data = pd.read_csv("20191201_20191220.csv", sep=',', nrows = row_num)
for i in range(row_num):
    sample_data = data.iloc[i][2].split(',')
    x = []
    y = []
    
    for index in range(1,len(sample_data)):
        sample = sample_data[index]
        sample = sample.split('[')[-1]
        position = sample.split(' ')[1:3]
        x.append(eval(position[0]))
        y.append(eval(position[1]))
    
    plt.plot(x, y)
    plt.xlabel('longitude')
    plt.ylabel('latitude')

plt.savefig('image_road.png')


plt.figure(2)
row_num = 16
data = pd.read_csv("20191201_20191220.csv", sep=',', nrows = row_num)
for i in range(row_num):
    sample_data = data.iloc[i][2].split(',')
    x = []
    y = []
    plt.figure(1)

    for index in range(1,len(sample_data)):
        sample = sample_data[index]
        sample = sample.split('[')[-1]
        position = sample.split(' ')[1:3]
        x.append(eval(position[0]))
        y.append(eval(position[1]))
    
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.plot(x, y)

plt.show()    
plt.savefig('collection_image_road.png')