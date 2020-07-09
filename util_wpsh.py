# -*- coding: utf-8 -*-
# @Time    : 2020/6/28 20:11
# @Author  : Mr zhou
# @FileName: util_wpsh.py
# @Software: PyCharm
# @weixin    ：dayinpromise1314
import os
from netCDF4 import Dataset
import numpy as np
import cv2

def load_data(path, datachoose):
    #读取lat,lon,speed,z,u,v,vo,t  2011050712 133 1254 18 100100.nc
    datasize = {}
    data_combine = {}
    for key in datachoose:
        datasize[key] = []
        data_combine[key] = []
    data_combine["lat"] = []
    data_combine["lon"] = []
    data_combine["speed"] = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ".nc" not in file:
                continue
            data_combine["lat"].append(int(file[10:13]) / 10)
            data_combine["lon"].append(int(file[13:17]) / 10)
            data_combine["speed"].append(int(file[17:19]))
    "输入维度为[ 时间，level，经度，维度。]"
    for root, dirs, files in os.walk(path):
        for key in datachoose:
            pool = ()
            for file in files:
                data = Dataset(path + file).variables
                npp = data[key][:]
                npp = cv2.normalize(npp, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                pool = pool + (npp,)
                data_combine[key] = np.concatenate(pool, axis=0)
    return data_combine#shape=({10},26,81,81,3)


def diff(input, x_l, y_l):
    output = np.zeros(shape=(x_l - 2, y_l - 2))
    for x in range(1, x_l - 1):
        for y in range(1, y_l - 1):
            output[x - 1, y - 1] = np.arctan((input[x, y + 1] - input[x, y - 1]) / (input[x + 1, y] - input[x - 1, y] + 1e-6))
    return output
