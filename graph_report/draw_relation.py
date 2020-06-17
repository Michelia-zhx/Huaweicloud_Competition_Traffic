import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_relation():
    data = pd.read_csv("Huaweicloud_Competition_Traffic/datasets/train_12_LiuXian_W2E.csv")
    use_num = 1000
    data = data[0:use_num]
    time = data['timestamp']
    speed = data['speed']
    tti = data['TTI']
    car_count = data['car_count']

    plt.figure(12)
    plt.subplot(121)
    plt.title("relation between TTI and average speed")
    plt.xlabel("average speed")
    plt.ylabel("TTI")
    plt.scatter(speed, tti, c='skyblue', alpha=0.7)
    plt.subplot(122)
    plt.title("relation between TTI and car number")
    plt.xlabel("car number")
    plt.ylabel("TTI")
    plt.scatter(car_count, tti, c='orange', alpha=0.7)
    plt.show()

draw_relation()