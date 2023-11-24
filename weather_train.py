import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary


def load_data(data_path):
    # 读取数据
    data = pd.read_csv(data_path)
    data['前一天'] = data['前一天'].astype(str).apply(
        lambda r: np.mean(list(map(float, re.sub(r'[\[\]]', '', r).split(',')))))
    data['上一年同一天'] = data['上一年同一天'].astype(str).apply(
        lambda r: np.mean(list(map(float, re.sub(r'[\[\]]', '', r).split(',')))))
    data['当天'] = data['当天'].astype(str).apply(
        lambda r: np.mean(list(map(float, re.sub(r'[\[\]]', '', r).split(',')))))
    # 数据预处理，防止异常数据出现，导致loss:nan出现
    data['前一天'].fillna(data['当天'], inplace=True)
    data['上一年同一天'].fillna(data['当天'], inplace=True)

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    data.plot(x='前一天', y='当天', kind='scatter')
    plt.ylabel("前一天ºC")
    plt.xlabel("当天ºC")
    plt.title('前一天/当天【气温】')
    data.plot(x='上一年同一天', y='当天', kind='scatter')
    plt.title('上一年同一天/当天【气温】')
    plt.ylabel("上一年同一天ºC")
    plt.xlabel("当天ºC")
    plt.show()

    # 提取特征和目标
    features = data[['前一天', '上一年同一天']]
    target = data['当天']

    # 划分训练集、验证集和测试集
    x_train, x_temp, y_train, y_temp = train_test_split(features, target, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    return x_train, x_val, x_test, y_train, y_val, y_test


def train(train_loader, val_loader, model, criterion, optimizer, num_epochs=100):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            # print('Input:', inputs)
            # print('Target:', targets)
            # print('Gradients:', [p.grad.data for p in model.parameters()])
            # print('Weights:', [p.data for p in model.parameters()])
            # print('Outputs:', outputs)
            if batch_idx % 10 == 0:
                print(f'Training - Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets.view(-1, 1)).item()

        val_loss /= len(val_loader)
        print(f'Validation - Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model.pth')


def visualize_model(model, data_loader):
    model.eval()
    inputs, targets = next(iter(data_loader))
    with torch.no_grad():
        outputs = model(inputs)

    if inputs.ndim > 1:
        num_features = inputs.shape[1]

        # 创建子图，每个特征对应一个子图
        fig, axs = plt.subplots(1, num_features, figsize=(15, 4))

        # 遍历每个特征，绘制散点图
        for i in range(num_features):
            axs[i].scatter(inputs[:, i].numpy(), targets.numpy(), label=f'特征 {i + 1}真实数据')
            axs[i].scatter(inputs[:, i].numpy(), outputs.numpy(), label=f'特征 {i + 1}预测结果')
            axs[i].set_xlabel(f'特征 {i + 1}')
            axs[i].set_ylabel('目标')
            axs[i].legend()

        plt.show()
    else:
        plt.scatter(inputs.numpy(), targets.numpy(), label='真实数据')
        plt.scatter(inputs.numpy(), outputs.numpy(), label='预测结果')
        plt.xlabel('Input')
        plt.ylabel('Target')
        plt.legend()
        plt.show()


# 定义神经网络模型
class TemperaturePredictor(nn.Module):
    def __init__(self, input_size=365, hidden_size=64, output_size=1):
        super(TemperaturePredictor, self).__init__()
        # 非线性关系
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # 线性关系
        # self.model = nn.Linear(input_size, output_size)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test = load_data('weather-2022_2021.csv')
    # 转换为 PyTorch 张量
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    # 创建数据加载器
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    input_size = len(x_train.columns)
    hidden_size = 64
    output_size = 1
    model = TemperaturePredictor(input_size, hidden_size, output_size)
    # 打印模型结构
    summary(model, input_size=(input_size,))
    # 训练模型
    train(train_loader, val_loader, model, model.criterion, model.optimizer, num_epochs=1000)
    # 加载模型参数
    model.load_state_dict(torch.load('model.pth'))
    # 可视化模型预测结果
    visualize_model(model, test_loader)
    # 模型信息
    print('模型信息：', model.state_dict())
