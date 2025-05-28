#下载和导入需要的库
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# math.radians(180)
# math.degrees(math.pi)
# 读取数据
def read_data(filename):
    f = open(filename, 'r')
    file = f.readlines()
    temp = []
    for r in range(len(file)):
        line = file[r].split()
        # line = line[-1]
        temp = pd.concat([pd.DataFrame(temp), pd.DataFrame([line])], axis=1) #拼接
    f.close()
    data_fun = np.array(temp).astype(float)
    return data_fun


cwd = os.getcwd()
basePath = cwd + "/data/"
rdata1 = read_data(basePath + "02pitch.txt").ravel()
rdata2 = read_data(basePath + "03roll.txt").ravel()

rdata1 = rdata1 * 180 / math.pi
rdata2 = rdata2 * 180 / math.pi


#添加roll pitch限制
# len = rdata1.size
# bench = 0
# for i in range(len):
#     if rdata1[i-bench] > 45 or rdata1[i-bench] < -45 or rdata2[i-bench] > 15 or rdata2[i-bench] < -15:
#         rdata1 = np.delete(rdata1, i-bench)
#         rdata2 = np.delete(rdata2, i-bench)
#         bench += 1


#绘制前两个特征的二维散点图
plt.scatter(rdata1, rdata2, alpha=0.1)
plt.xlabel('pitch (deg)')  # 横坐标轴标题
plt.ylabel('roll (deg)')  # 纵坐标轴标题
plt.show()