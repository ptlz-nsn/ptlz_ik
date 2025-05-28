import torch
import numpy as np
import os 
cwd = os.getcwd()

def read_data_joint_joint():
  basePath = cwd +"/data/"

  data1 = np.loadtxt(basePath + "joint1.txt")

  data2 = np.loadtxt(basePath + "joint2.txt")

  x = data1[1:, 0]
  y = data1[0, 1:]

  data1 = data1[1:, 1:]
  data2 = data2[1:, 1:]
  inputs = []
  outputs = []

  for i in range(x.size):
    for j in range(y.size):
        input = [data1[i, j], data2[i, j]]
        output = [x[i], y[j]]
        inputs.append(input)
        outputs.append(output)
  xs = torch.tensor(inputs, dtype=torch.float32)
  ys = torch.tensor(outputs, dtype=torch.float32)
  return xs, ys

def read_data_linear_euler():
  basePath = cwd +"/data/"
  inputs = np.loadtxt(basePath + "linear.txt")
  outputs = np.loadtxt(basePath + "euler.txt")
  xs = torch.tensor(inputs, dtype=torch.float32)
  ys = torch.tensor(outputs, dtype=torch.float32)
  return xs, ys

x, y = read_data_linear_euler()

# 保存多个张量（字典形式）
data = {
    'x': x,
    'y': y
}

basePath = cwd +"/net/"
torch.save(data, basePath + "train_tensors.pt")
