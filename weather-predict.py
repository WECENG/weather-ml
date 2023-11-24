# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "天气预测"
__Created__ = 2023/11/24 17:19
"""
import torch

from weather_train import TemperaturePredictor


def predict(pre_weather, last_weather, model):
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # 将输入转换为 PyTorch 张量
    inputs = torch.tensor([[pre_weather, last_weather]], dtype=torch.float32)

    with torch.no_grad():
        output = model(inputs)

    return output.item()


if __name__ == '__main__':
    # 初始化模型
    model = TemperaturePredictor(input_size=2)
    # 预测
    print('预测结果：', predict(23.0, 25.0, model))
