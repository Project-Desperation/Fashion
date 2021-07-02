import os, time, math
import torch
from torch import optim, nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device: {}.".format(str(device)))


# -----------------------------------------------------------------------------
## 数据预处理
# data load
def load_data(path):
    raw_data = []
    with open(path, 'r') as f:
        lines = f.readlines()[:120]
        for line in lines:
            attrs_num = line.rstrip().split(': ')[1].split(' ')
            attrs_num = [int(x) for x in attrs_num]
            raw_data.append(attrs_num)
    all_data = np.array(raw_data, dtype='float32')
    all_data = all_data.T
    return all_data


def normalize(all_data):
    trend_data = []
    for data in all_data:
        row = []
        for i in range(len(data) - days_look_back):
            avg = 0
            for j in range(days_look_back):
                avg += data[i + j]
            avg /= days_look_back
            row.append((data[i + days_look_back] - avg) / avg)
        trend_data.append(row)
    trend_data = np.array(trend_data, dtype='float32')
    return trend_data


# # 均值化
# min_trend = trend_data.min(axis=1)
# max_trend = trend_data.max(axis=1)
# trend_norm = ((trend_data + np.reshape(min_trend, (attr_num, 1))) / np.reshape(-min_trend, (attr_num, 1))) + 1
# with open(os.path.join(PATH, f'min_trend.txt'), 'w') as f:
#   f.write(str(list(min_trend)))
#   f.close()
# with open(os.path.join(PATH_BEST, f'min_trend.txt'), 'w') as f:
#   f.write(str(list(min_trend)))
#   f.close()
# # 反均值化函数（展示用）
# def denorm(data, minimum):
#   return (data - 1) * (-minimum) - minimum

# create_dataset
def create_dataset(data, days_for_train=5, days_to_predict=1) -> (np.array, np.array):  # 更改days_to_predict可以预测N天后的
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train - days_to_predict + 1):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append([])
        for j in range(days_to_predict):
            dataset_y[i].append(data[i + days_for_train + j])
    return (np.array(dataset_x), np.array(dataset_y).T)


# -----------------------------------------------------------------------------
# LSTM预测
# practice lstm of pytorch
class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):  # 【output_size：输出序列长度可以改变】
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        # x = self.dropout(x)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # 模型参数
    DAYS_FOR_TRAIN = 10
    DAYS_TO_PREDICT = 7
    NUM_LAYERS = 2
    HIDDEN_SIZE = 8
    TRAIN_SIZE = 90

    epochs = 500
    lr = 1e-2

    attributes = ['floral', 'striped', 'plaid', 'leopard', 'camo', 'graphic',
                  'crew neck', 'square neck', 'v-neck',
                  'maxi dress', 'midi dress', 'mini dress',
                  'denim', 'knit', 'faux leather', 'cotton', 'chiffon', 'satin',
                  'mesh', 'ruched', 'cutout', 'lace', 'frayed', 'wrap',
                  'tropical', 'peasant', 'swim', 'bikini', 'active', 'cargo']

    attr_num = len(attributes)
    # input data path
    input_path = 'data/lstm-label.txt'
    days_look_back = 5
    output_size = 1
    name = 'raw'

    # 过程模型保存路径
    PATH = f'model_test/model_{name}_{days_look_back}_{DAYS_TO_PREDICT}_{DAYS_FOR_TRAIN}{NUM_LAYERS}{HIDDEN_SIZE}'
    # 最优结果保存路径
    PATH_BEST = f'model_test/model_{name}_{days_look_back}_{DAYS_TO_PREDICT}_{DAYS_FOR_TRAIN}{NUM_LAYERS}{HIDDEN_SIZE}_best'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    if not os.path.exists(PATH_BEST):
        os.makedirs(PATH_BEST)

    trend_norm = load_data(input_path)

    for day in range(DAYS_TO_PREDICT):
        attr_train_result, attr_test_result = dict(), dict()
        for index in range(len(attributes)):
            data_x, data_y = create_dataset(trend_norm[index], days_for_train=DAYS_FOR_TRAIN,
                                            days_to_predict=DAYS_TO_PREDICT)
            train_x = torch.from_numpy(data_x[:TRAIN_SIZE].reshape(-1, 1, DAYS_FOR_TRAIN))
            train_y = torch.from_numpy(data_y[day][:TRAIN_SIZE].reshape(-1, 1, 1))
            test_x = torch.from_numpy(data_x[TRAIN_SIZE:].reshape(-1, 1, DAYS_FOR_TRAIN))
            test_y = torch.from_numpy(data_y[day][TRAIN_SIZE:].reshape(-1, 1, 1))

            model = LSTM_Regression(DAYS_FOR_TRAIN, HIDDEN_SIZE, output_size=1, num_layers=NUM_LAYERS).to(device)
            loss_function = nn.MSELoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            attr_train_result[attributes[index]] = []
            attr_test_result[attributes[index]] = []
            for i in range(epochs):
                model.train()
                out = model(train_x.to(device))
                loss = loss_function(out, train_y.to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (i + 1) % 10 == 0:
                    model.eval()
                    test_out = model(test_x.to(device))
                    test_loss = loss_function(test_out, test_y.to(device))
                    if (i + 1) % 100 == 0:
                        print('Attribute: {}, Epoch: {}, Loss:{:.5f}, Test Loss:{:.5f}'.format(attributes[index], i + 1,
                                                                                               loss.item(),
                                                                                               test_loss.item()))
                    torch.save(model,
                               os.path.join(PATH, f'{attributes[index]}_day{day}_epoch_{i + 1}.pt'))
                    attr_train_result[attributes[index]].append(loss.item())
                    attr_test_result[attributes[index]].append(test_loss.item())

        for k, v in attr_test_result.items():
            print(k, [round(x, 4) for x in v])

        print("Model choosing: day{}".format(day))
        for index in range(30):
            epoch = attr_test_result[attributes[index]].index(min(attr_test_result[attributes[index]]))
            epoch = (epoch + 1) * 10
            if epoch < 70:
                epoch = 200
            print('Attribute: {}, BEST_Epoch: {}'.format(attributes[index], epoch))
            with open(os.path.join(PATH, f'{attributes[index]}_day{day}_epoch_{epoch}.pt'), 'rb') as src:
                with open(os.path.join(PATH_BEST, f'{attributes[index]}_day{day}.pt'), 'wb') as des:
                    des.write(src.read())

        with open(os.path.join(PATH, f'attr_result_day{day}.txt'), 'w') as f:
            f.write(str((attr_train_result, attr_test_result)))
        with open(os.path.join(PATH_BEST, f'attr_result_day{day}.txt'), 'w') as f:
            f.write(str((attr_train_result, attr_test_result)))
