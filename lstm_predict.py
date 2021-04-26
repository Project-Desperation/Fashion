import os

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn

matplotlib.use('Agg')

from datetime import timedelta
from dateutil import parser
from copy import copy


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
        x = self.dropout(x)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x


# -----------------------------------------------------------------------------
class DynamicPredictor():
    '''
    动态预测器
    '''

    def __init__(self, DAYS_FOR_TRAIN, DAYS_TO_PREDICT, DAYS_LOOK_BACK, PATH, attribute_name, init_data, start_date):
        super().__init__()
        self.DAYS_FOR_TRAIN = DAYS_FOR_TRAIN
        self.DAYS_TO_PREDICT = DAYS_TO_PREDICT
        self.DAYS_LOOK_BACK = DAYS_LOOK_BACK
        self.NUM_LAYERS = 2
        self.HIDDEN_SIZE = 8
        self.PATH = PATH

        self.today = parser.parse(start_date) + timedelta(days=len(init_data) - 1)
        self.day = len(init_data)
        self.attribute_name = attribute_name
        self.raw_data = init_data
        self.trend_data = []
        for i in range(len(init_data) - self.DAYS_LOOK_BACK):
            avg = 0
            for j in range(self.DAYS_LOOK_BACK):
                avg += init_data[i + j]
            avg /= self.DAYS_LOOK_BACK
            self.trend_data.append((init_data[i + self.DAYS_LOOK_BACK] - avg) / avg)

        self.pred_model = [
            LSTM_Regression(self.DAYS_FOR_TRAIN, self.HIDDEN_SIZE, output_size=1, num_layers=self.NUM_LAYERS) for i in
            range(self.DAYS_TO_PREDICT)]
        for i in range(self.DAYS_TO_PREDICT):
            self.pred_model[i].load_state_dict(torch.load(os.path.join(self.PATH, f'{self.attribute_name}_day{i}.pth')))
            self.pred_model[i].cuda()

        self.predict_data = None
        self.predict()
        # self.predict_data = [0 for i in range(self.DAYS_TO_PREDICT-1)]
        # for i in range(len(self.trend_data)-self.DAYS_FOR_TRAIN+1):
        #   self.append_predict(self.trend_data[i:(i+self.DAYS_FOR_TRAIN)])

    def append_data(self, data):
        data = float(data)
        # 相对化
        avg = 0
        for i in range(self.DAYS_LOOK_BACK):
            avg += self.raw_data[-i - 1]
        avg /= self.DAYS_LOOK_BACK
        self.trend_data.append((data - avg) / avg)
        self.raw_data.append(data)
        self.today = self.today + timedelta(days=1)
        self.day += 1
        # self.append_predict(self.trend_data[-self.DAYS_FOR_TRAIN:])
        self.predict()

    def predict(self):
        data = []
        for i in range(len(self.trend_data) - self.DAYS_FOR_TRAIN + 1):
            _x = self.trend_data[i:(i + self.DAYS_FOR_TRAIN)]
            data.append(_x)
        data = torch.tensor(data).reshape(-1, 1, self.DAYS_FOR_TRAIN).cuda()
        self.pred_model[0].eval()
        self.predict_data = self.pred_model[0](data).cpu().reshape(-1).detach().numpy().tolist()
        for i in range(1, self.DAYS_TO_PREDICT):
            self.pred_model[i].eval()
            pred_test = self.pred_model[i](data[-1:])  # 全量训练集的模型输出 (seq_size, batch_size, output_size)
            pred_test = float(pred_test.cpu().view(-1).data.numpy()[0])
            self.predict_data.append(pred_test)

    # def append_predict(self, data_x):
    #   self.predict_data.append(0)
    #   data_x = torch.tensor(data_x, dtype=torch.float32).reshape(-1, 1, self.DAYS_FOR_TRAIN)
    #   for j in range(self.DAYS_TO_PREDICT):
    #     self.pred_model[j].eval()
    #     pred_test = self.pred_model[j](data_x) # 全量训练集的模型输出 (seq_size, batch_size, output_size)
    #     pred_test = float(pred_test.cpu().view(-1).data.numpy()[0])
    #     self.predict_data[-self.DAYS_TO_PREDICT+j] = pred_test

    def plot_prediction(self, trend_len=None):
        plt.figure(figsize=(16, 4))
        if not trend_len:
            trend_len = len(self.trend_data)
        predict_data = copy(self.predict_data[-self.DAYS_TO_PREDICT:])
        predict_data.insert(0, self.trend_data[-1])
        plt.plot(range(trend_len - 1, trend_len + self.DAYS_TO_PREDICT), predict_data, 'r', label='prediction')
        for x, y in zip(range(trend_len - 1, trend_len + self.DAYS_TO_PREDICT), np.array(predict_data)*1.01):
            plt.text(x, y, '{:.2f}'.format(y/1.01))
        plt.plot(self.trend_data[-trend_len:], 'b', label='real')
        plt.plot((trend_len - 1, trend_len - 1), (-0.2, 0.2), 'g--')
        plt.legend(loc='best')
        plt.title('Prediction of {}'.format(self.attribute_name.upper()))
        plt.ylabel("Change Rate to {}-day Avg".format(self.DAYS_LOOK_BACK))
        plt.xlabel('Day')
        plt.savefig("static/img/test_prediction.jpg", bbox_inches='tight')  # 【修改此处以控制plt的输出位置】
        # plt.show()  # 【修改此处以控制plt的输出位置】
        return self.today.strftime('%Y-%m-%d')

    def plot_validation(self, loss_function=nn.MSELoss()):
        plt.figure()
        plt.plot(range(self.DAYS_FOR_TRAIN, len(self.trend_data)), self.predict_data[:-self.DAYS_TO_PREDICT], 'r',
                 label='prediction')
        plt.plot(self.trend_data, 'b', label='real')
        plt.legend(loc='best')
        plt.title('Validation of {}'.format(self.attribute_name.upper()))

        if len(range(self.DAYS_FOR_TRAIN, len(self.trend_data))) == 0:
            loss = -1
        else:
            loss = loss_function(torch.tensor(self.predict_data[:-self.DAYS_TO_PREDICT]),
                                 torch.tensor(self.trend_data[self.DAYS_FOR_TRAIN:]))
            loss = loss.detach().numpy()
        plt.ylabel("Change Rate to {}-day Avg".format(self.DAYS_LOOK_BACK))
        plt.xlabel('Day\nMSE Loss: {:.6f}'.format(loss))
        plt.savefig("static/img/test_validation.jpg", bbox_inches='tight')  # 【修改此处以控制plt的输出位置】
        # plt.show()  # 【修改此处以控制plt的输出位置】
        return self.today.strftime('%Y-%m-%d'), loss


# -----------------------------------------------------------------------------
# 主程序
DAYS_FOR_TRAIN = 10
DAYS_TO_PREDICT = 7
NUM_LAYERS = 2
HIDDEN_SIZE = 8
DAYS_LOOK_BACK = 5
PATH = 'model/model5_1028_7R_best/'
input_path = 'data/lstm-label.txt'
attributes = ['floral', 'striped', 'plaid', 'leopard', 'camo', 'graphic',
              'crew neck', 'square neck', 'v-neck',
              'maxi dress', 'midi dress', 'mini dress',
              'denim', 'knit', 'faux leather', 'cotton', 'chiffon', 'satin',
              'mesh', 'ruched', 'cutout', 'lace', 'frayed', 'wrap',
              'tropical', 'peasant', 'swim', 'bikini', 'active', 'cargo']

## 数据预处理
# 【加载预置数据all_data，是一个属性*时间的array，all_data[0][0]表示第0种属性（floral）在第0天的出现次数】
raw_data = []
with open(input_path, 'r') as f:
    lines = f.readlines()[:120]
    for line in lines:
        attrs_num = line.rstrip().split(': ')[1].split(' ')
        atrrs_num = [int(x) for x in attrs_num]
        raw_data.append(attrs_num)
all_data = np.array(raw_data, dtype='float32')
all_data = all_data.T

# 【生成predictor_dict，predictor_dict['floral']指向对floral属性进行预测的预测器，预测器的初始输入是15天（all_data[index][:15]）】
predictor_dict = {}
for index in range(len(attributes)):
    predictor_dict[attributes[index]] = DynamicPredictor(DAYS_FOR_TRAIN, DAYS_TO_PREDICT, DAYS_LOOK_BACK, PATH,
                                                         attributes[index],
                                                         all_data[index].tolist()[:DAYS_FOR_TRAIN + DAYS_LOOK_BACK],
                                                         "2020-06-18")

plt.style.use('ggplot')


# print(predictor_dict['floral'].today)

# 【这个函数调用一次更新一天】
def daily_update(predictor_dict, data, attributes):
    for index in range(len(attributes)):
        key = attributes[index]
        day = predictor_dict[key].day
        predictor_dict[key].append_data(data[index].tolist()[day])
        index += 1


# 测试时先循环了10天，可以不循环
for _ in range(40):
    daily_update(predictor_dict, all_data, attributes)

# 【绘图部分分成两部分，预测和评估】
# 【attributes[index]就是当前要绘制的属性】
index = 11
# DynamicPredictor.plot_prediction()方法绘制预测的图像，返回值就是日期（string形式） 2020-08-19
date = predictor_dict[attributes[index]].plot_prediction(
    trend_len=min(len(predictor_dict[attributes[index]].trend_data), 30))
# DynamicPredictor.plot_validation()方法绘制评估的图像，返回值是日期（string形式）和当前预测的均方误差（MSELoss） 2020-08-19 0.036840852
date, loss = predictor_dict[attributes[index]].plot_validation()
print(date, loss)
