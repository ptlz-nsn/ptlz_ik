#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <iostream>
#include <cstring>  // 提供memcpy函数
int main() {
    auto interpreter = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile("/home/host/code/python/palms/net/forward_net.mnn"));
    
    MNN::ScheduleConfig config;
    auto session = interpreter->createSession(config);
    
    // auto input_tensor = interpreter->getSessionInput(session, "input_2d");
    // auto output = interpreter->getSessionOutput(session, "output_2d");
    // 2. 获取输入输出名称（关键修改）
    const std::string input_name = "input_2d";   // 与ONNX导出时的input_names一致
    const std::string output_name = "output_2d"; // 与ONNX导出时的output_names一致
    
    // 3. 准备输入数据 (batch=1, dim=2)
    float input_data[2] = {137.7f, 137.7f};
    MNN::Tensor* input_tensor = interpreter->getSessionInput(session, input_name.c_str());
    
    // 4. 填充输入数据
    std::memcpy(input_tensor->host<float>(), input_data, 2 * sizeof(float));
    input_tensor->copyFromHostTensor(input_tensor);
    
    // 5. 执行推理
    interpreter->runSession(session);

    // 6. 获取输出
    MNN::Tensor* output_tensor = interpreter->getSessionOutput(session, output_name.c_str());
    MNN::Tensor host_tensor(output_tensor, MNN::Tensor::CAFFE);
    output_tensor->copyToHostTensor(&host_tensor);

    // 7. 读取结果
    const float* output_data = host_tensor.host<float>();
    std::cout << "Output: [" 
              << output_data[0] << ", " 
              << output_data[1] << "]" << std::endl;

    return 0;
}