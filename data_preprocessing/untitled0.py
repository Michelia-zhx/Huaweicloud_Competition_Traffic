# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:47:16 2020

@author: zhanghanxiao
"""

import pandas as pd

ps = pd.read_csv("../../20191201_20191220.csv", iterator = True);
while(1):
    print(1)
    try:
        data = ps.get_chunk(10)
    except:
        break