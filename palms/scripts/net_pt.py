import scipy.io as scio  # 需要用到scipy库
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, hidden_size=64):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)  # 输入层 → 隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)   # 隐藏层 → 输出层
        self.activation = nn.ReLU()            # 激活函数

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # 最后一层不加激活（适合回归任务）
        return x



loaded_data = torch.load('train_tensors.pt')
X = loaded_data['X']
Y = loaded_data['Y']

# 初始化
model = Net()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 数据划分
train_X, test_X = X[:800], X[800:]
train_Y, test_Y = Y[:800], Y[800:]

# 训练循环
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(train_X)
    loss = criterion(outputs, train_Y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


# 方法1：通过追踪(tracing)
scripted_model = torch.jit.trace(model, X)
scripted_model.save('model_scripted.pt')

model.eval()  # 必须设置为评估模式

# 生成2D虚拟输入（batch_size=1, input_dim=2）
dummy_input = torch.randn(1, 2)  # 注意维度匹配

# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    "forward_net.onnx",
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