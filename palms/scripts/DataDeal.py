import scipy.io as scio  # 需要用到scipy库
import numpy as np

basePath = "./src/parallel_ankle/data/"

data1 = np.loadtxt(basePath + "joint1.txt")

data2 = np.loadtxt(basePath + "joint2.txt")

x = data1[1:, 0]
y = data1[0, 1:]

data1 = data1[1:, 1:]
data2 = data2[1:, 1:]
inputs = []
xindex = 0
outputs = []
for i in range(x.size):
    for j in range(y.size):
        input = [data1[i, j], data2[i, j]]
        output = [x[i], y[j]]
        inputs.append(input)
        outputs.append(output)
        

savePath = "./src/parallel_ankle/matlab/"
# 保存到当前路径下
scio.savemat(savePath + 'inputs.mat', {'inputs':inputs}) 
scio.savemat(savePath + 'outputs.mat', {'outputs':outputs}) 



