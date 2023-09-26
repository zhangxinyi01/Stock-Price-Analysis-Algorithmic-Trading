import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from data_process import split_time_series_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
print("当前文件的路径是:", current_directory)

def normalize_data(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val

def denormalize_data(scaled_data, min_val, max_val):
    denormalized_data = scaled_data * (max_val - min_val) + min_val
    return denormalized_data

x_train, y_trainn, x_test, y_test = split_time_series_data(pd.read_excel(current_directory+'/stock_history.xlsx'),train_ratio=0.75) 
y_trainn, min_vall, max_vall = normalize_data(y_trainn)
y_test, min_val, max_val = normalize_data(y_test)
x_train = torch.tensor(x_train, dtype=torch.float32).to(device) 
df = pd.DataFrame()
cnt = 0
for data in y_trainn: 
    cnt = cnt + 1
    print(cnt)
    y_train = torch.tensor(list(y_trainn[:][data]), dtype=torch.float32).to(device) 
    
        # 定义一个简单的LSTM模型
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x.unsqueeze(0).unsqueeze(2))
            output = self.fc(lstm_out[:, -1, :])
            return output

    # 创建模型实例
    input_size = 1  # 输入特征的维度，因为 x_train 是一维的
    hidden_size = 10  # LSTM隐藏层的单元数量
    output_size = 1  # 输出维度，因为 y_train 是一维的
    model = SimpleLSTM(input_size, hidden_size, output_size).to(device) 

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 100

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 使用模型进行预测
    predictions = []
    
    time_steps = len(x_train)-len(x_test)
    for i in range(len(x_train) - time_steps):
        # 从输入序列中提取时间步长窗口
        x = x_train[i:i + time_steps]
        prediction = model(x)
        pre = denormalize_data(prediction.item(), min_val[:][data], max_val[:][data])
        predictions.append(pre)
    df[data] = predictions
df.to_csv("pred_res.csv")
excel_file = "output.xlsx"
df.to_excel(excel_file, index=True) 
