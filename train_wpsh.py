# -*- coding: utf-8 -*-
# @Time    : 2020/6/28 19:59
# @Author  : Mr zhou
# @FileName: train_wpsh.py
# @Software: PyCharm
# @weixin    ：dayinpromise1314
import numpy as np
import pandas as pd
import os
from util_wpsh import load_data
from network import Network
import matplotlib.pyplot as plt
# %% data_input
lookback = 4  # lookback在这里可以画一个图，比较一下取不同的lookback
lookforward = 1  # 预测多少个时间步 之后的数据。
datachoose = ['z', 't', 'u', 'v']
data_file = []
path = 'G:/ECMWFdata_sth/'
for root, dirs, files in os.walk(path):
    for dir in dirs:
        data_combine = load_data(path + dir + "/", datachoose)  # shape=[batch,4,301,273]
        t_l = data_combine['z'].shape[0]
        time = []
        for i in range(0, len(files)):
            time.append(files[i][0:8])
        data_file.append({"input": data_combine, "len": t_l, 'time': time})
# %% 读取对应的时间的WPSH index
path = r'C:/Users/Dell/WPSH_index.csv'
s_time = []
s_area = []
s_intensity = []
s_ridge = []
s_wp = []
with open(os.path.join(path), encoding='utf-8') as rf:
    file_data = pd.read_csv(rf, header=None)
    a = file_data  # iloc就可以和matlab一样引用了
    s1 = list(a.iloc[:, 0])
    s2 = list(a.iloc[:, 1])
    s3 = list(a.iloc[:, 2])
    s4 = list(a.iloc[:, 3])
    s5 = list(a.iloc[:, 4])
    s6 = list(a.iloc[:, 5])
    s7 = list(a.iloc[:, 6])
    for i in range(len(a)):
        s_time.append(str(s1[i]) + str(s2[i]).zfill(2) + str(s3[i]).zfill(2))
        s_area.append(str(s4[i]).zfill(3))
        s_intensity.append(str(s5[i]).zfill(3))
        s_ridge.append(str(s6[i]).zfill(2))
        s_wp.append(str(s7[i]).zfill(3))
# %%label
path = 'G:/ECMWFdata_sth/'
area = []
intensity = []
ridge = []
wp = []
for root, dirs, files in os.walk(path):
    for file in files:
        if ".nc" not in file:
            continue
        file_time = file[0:8]
        for i in range(len(a)):
            if file_time == s_time[i]:
                area.append(s_area[i])
                intensity.append(s_intensity[i])
                ridge.append(s_ridge[i])
                wp.append(s_wp[i])
                break
area=[float(area[i]) for i in range(len(area))]
intensity=[float(intensity[i]) for i in range(len(intensity))]
ridge=[float(ridge[i]) for i in range(len(ridge))]
wp=[float(wp[i]) for i in range(len(wp))]
wp=[np.mean(wp[wp!=-9999]) if value == -9999 else value for value in wp]
# %% train_modelv
# 选择文件夹里所有的数据
pool1 = ()
input_data = []
index_file = []
data_key = ['z', 't', 'u', 'v']
k = 0
for num in range(int(len(data_file) / 20)):  # 选择文件夹
    data_file_input = data_file[num]["input"]  # list len=10, array shape=(4,num_files,301,273)
    # 转换数据
    i = 0
    data_array = np.zeros([data_file_input['z'].shape[0], len(data_key), 301, 273])  # eg. shape=(17,4,301,273,1)
    for key1 in data_key:
        data_array[:, i, :] = data_file_input[key1][:]
        i = i + 1
    data_array = np.expand_dims(data_array, axis=-1)
    pool1 = pool1 + (data_array,)
    index_file = index_file + list(
        range(k, data_file_input['z'].shape[0] -lookback + k))  # index_1 从这里选择序号
    k = k + data_file_input['z'].shape[0]

# 归一化
from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()
input_data = np.concatenate(pool1, axis=0)
for i in range(input_data.shape[0]):
    for j in range(input_data.shape[1]):
        for k in range(input_data.shape[4]):
            input_data[i, j, :, :, k] = std_scaler.fit_transform(input_data[i, j, :, :, k])
            
             
# 求出输入和label
index_1 = index_file
index_2 = [i + 1 for i in index_1]
index_3 = [i + 2 for i in index_1]
index_4 = [i + 3 for i in index_1]
index_5 = [i + 4 for i in index_1]
input_data_z=np.zeros([len(index_1),4,301,273,1])
input_data_sst=np.zeros([len(index_1),4,301,273,1])
input_data_u=np.zeros([len(index_1),4,301,273,1])
input_data_v=np.zeros([len(index_1),4,301,273,1])

for j in range(len(index_1)):
    
    input_data_z[j,0,:]=input_data[index_1[j],0,:]
    input_data_z[j,1,:]=input_data[index_2[j],0,:]
    input_data_z[j,2,:]=input_data[index_3[j],0,:]
    input_data_z[j,3,:]=input_data[index_4[j],0,:]
    
    input_data_sst[j,0,:]=input_data[index_1[j],1,:]
    input_data_sst[j,1,:]=input_data[index_2[j],1,:]
    input_data_sst[j,2,:]=input_data[index_3[j],1,:]
    input_data_sst[j,3,:]=input_data[index_4[j],1,:]
    
    input_data_u[j,0,:]=input_data[index_1[j],2,:]
    input_data_u[j,1,:]=input_data[index_2[j],2,:]
    input_data_u[j,2,:]=input_data[index_3[j],2,:]
    input_data_u[j,3,:]=input_data[index_4[j],2,:]
    
    input_data_v[j,0,:]=input_data[index_1[j],3,:]
    input_data_v[j,1,:]=input_data[index_2[j],3,:]
    input_data_v[j,2,:]=input_data[index_3[j],3,:]
    input_data_v[j,3,:]=input_data[index_4[j],3,:]


input = [input_data_z, input_data_sst, input_data_u, input_data_v]
# label = np.array(
#     [np.array([float(wp[i]) for i in range(len(index_5))]),])
label = np.array(
    [np.array([float(area[index_5[i]]) for i in range(len(index_5))]), np.array([float(intensity[index_5[i]]) for i in range(len(index_5))]),
     np.array([float(ridge[index_5[i]]) for i in range(len(index_5))]), np.array([float(wp[index_5[i]]) for i in range(len(index_5))])])
label=label.swapaxes(0,1)
# %%训练
model = Network()
history = model.fit(input, label, epochs=5,batch_size=2, validation_split=0.3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('./test2.jpg')
model.save('my_model_2.h5')
 # %%加载模型并且预测
from keras.models import load_model
model = load_model('my_model_2.h5')  #选取自己的.h模型名称
input = [input_data_z[5:7], input_data_sst[5:7], input_data_u[5:7], input_data_v[5:7]]
predict=model.predict(input)

# %%可视化
from keras.models import Model
layer_index=17 # the index of the layer you want to observe
intermediate_layer_model=Model(inputs=model.input, outputs=model.get_layer(index=layer_index).output)
intermediate_output = intermediate_layer_model.predict([input_data[index_1], input_data[index_2], input_data[index_3], input_data[index_4]])
layer_output = intermediate_output[0]
from pylab import *
for i in range(4):
    subplot(1,5,i+1)
    plt.imshow(layer_output[0,:,:,i],cmap='jet')
    if i==3:
        subplot(1,5,i+2)
plt.imshow(layer_output[0,:,:,i],cmap='jet')

plt.savefig('3.png')


