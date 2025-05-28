import scipy.io as scio  # 需要用到scipy库
# import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os 

class Net(nn.Module):
    def __init__(self, train_x:torch.Tensor, train_y:torch.Tensor, hidden_size=64):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)  # 输入层 → 隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)   # 隐藏层 → 输出层
        self.activation = nn.ReLU()            # 激活函数
        self.input_mean, self.input_std = train_x.mean(), train_x.std()
        self.output_mean, self.output_std = train_y.mean(), train_y.std()
    def forward(self, x):
        x = (x - self.input_mean) / self.input_std 
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # 最后一层不加激活（适合回归任务）
        x = x * self.output_std + self.output_mean 
        return x

cwd = os.getcwd()
basePath = cwd +"/net/"

loaded_data = torch.load(basePath + 'train_tensors.pt')
x = loaded_data['x']
y = loaded_data['y']

# 初始化
model = Net(train_x = x, train_y = y)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.01)


test_len = math.ceil(x.shape[0]*0.3)
test_start = math.ceil(x.shape[0]*0.5) - math.ceil(x.shape[0]*0.15)

# 数据划分
train_x, test_x = x, x[test_start:test_start+test_len]
train_y, test_y = y, y[test_start:test_start+test_len]

# 训练循环
losses = []
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(train_x)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')



model.eval()  # 必须设置为评估模式

# 生成2D虚拟输入（batch_size=1, input_dim=2）
dummy_input = torch.randn(1, 2)  # 注意维度匹配

# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    basePath+"forward_net.onnx",
    input_names=["input_2d"],
    output_names=["output_2d"],
    dynamic_axes={
        'input_2d': {0: 'batch'},  # 动态batch维度
        'output_2d': {0: 'batch'}
    },
    opset_version=13  # 推荐使用opset 13+
)

print("ONNX导出成功！输入输出维度：")
print(f"输入: {dummy_input.shape} -> 输出: {model(dummy_input).shape}")

# 绘制损失曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

with torch.no_grad():
    pred_Y = model(test_x)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(test_x[:, 0], test_x[:, 1], c='b', label='Input')
axes[0].set_title('Input Space')
axes[1].scatter(test_y[:, 0], test_y[:, 1], c='r', label='True Output')
axes[1].scatter(pred_Y[:, 0], pred_Y[:, 1], c='g', alpha=0.3, label='Predicted')
axes[1].set_title('Output Space')
plt.legend()
plt.show()

