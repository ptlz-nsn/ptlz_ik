import numpy as np
import onnxruntime as ort
import os
import math
cwd = os.getcwd()
basePath = cwd +"/net/"
# 创建 ONNX Runtime 会话
ort_session = ort.InferenceSession(basePath + "forward_net.onnx")

# 准备输入数据（需与 ONNX 模型的输入形状和类型匹配）
input_name = ort_session.get_inputs()[0].name  # 获取输入节点名称
input_shape = ort_session.get_inputs()[0].shape[1]  # 获取输入形状
print(input_shape)
dummy_input = np.random.randn(1, 2).astype(np.float32)  # 示例输入
dummy_input[0, 0] = 137.7
dummy_input[0, 1] = 137.7
# 运行推理
outputs = ort_session.run(
    None,  # 输出节点名称（None 表示全部输出）
    {input_name: dummy_input}
)
outputs = np.array(outputs)
outputs = outputs.ravel()
print("推理结果:", outputs)
print("推理结果:", outputs*180/math.pi)