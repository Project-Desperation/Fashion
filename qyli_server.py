from flask import Flask
from flask import render_template, url_for, request, jsonify
import urllib.request as urllib
import torch, torchvision
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import json
import os
import requests
# from google.colab.patches import cv2_imshow

import os, time, math, shutil, random

import mxnet as mx
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn, data as gdata
from mxnet.gluon.model_zoo import vision as models

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

from datetime import timedelta
from dateutil import parser
from copy import copy

# ----------------------------------------------------------------------------------------------------------------------
# LSTM预测
# practice lstm of pytorch
class LSTM_Regression(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):  # output_size：输出序列长度可以改变
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        x = self.dropout(x)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x


# ----------------------------------------------------------------------------------------------------------------------
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
        self.trend_norm = []
        for i in range(len(init_data) - self.DAYS_LOOK_BACK):
            avg = 0
            for j in range(self.DAYS_LOOK_BACK):
                avg += init_data[i + j]
            avg /= self.DAYS_LOOK_BACK
            self.trend_data.append(init_data[i + self.DAYS_LOOK_BACK] - avg)
            self.trend_norm.append((init_data[i + self.DAYS_LOOK_BACK] - avg) / avg)

        self.pred_model = [
            LSTM_Regression(self.DAYS_FOR_TRAIN, self.HIDDEN_SIZE, output_size=1, num_layers=self.NUM_LAYERS) for i in
            range(self.DAYS_TO_PREDICT)]
        for i in range(self.DAYS_TO_PREDICT):
            self.pred_model[i].load_state_dict(torch.load(os.path.join(self.PATH, f'{self.attribute_name}_day{i}.pth')))
            self.pred_model[i].cuda()

        self.predict_data = None
        self.predict()

    def append_data(self, data):
        data = float(data)
        # 相对化
        avg = 0
        for i in range(self.DAYS_LOOK_BACK):
            avg += self.raw_data[-i - 1]
        avg /= self.DAYS_LOOK_BACK
        self.trend_data.append(data - avg)
        self.trend_norm.append((data - avg) / avg)
        self.raw_data.append(data)
        self.today = self.today + timedelta(days=1)
        self.day += 1
        # self.append_predict(self.trend_norm[-self.DAYS_FOR_TRAIN:])
        self.predict()


    def predict(self):
        data = []
        for i in range(len(self.trend_norm) - self.DAYS_FOR_TRAIN + 1):
            _x = self.trend_norm[i:(i + self.DAYS_FOR_TRAIN)]
            data.append(_x)
        data = torch.tensor(data).reshape(-1, 1, self.DAYS_FOR_TRAIN).cuda()
        self.pred_model[0].eval()
        self.predict_data = self.pred_model[0](data).cpu().reshape(-1).detach().numpy().tolist()
        for i in range(1, self.DAYS_TO_PREDICT):
            self.pred_model[i].eval()
            pred_test = self.pred_model[i](data[-1:])  # 全量训练集的模型输出 (seq_size, batch_size, output_size)
            pred_test = float(pred_test.cpu().view(-1).data.numpy()[0])
            self.predict_data.append(pred_test)

    # 趋势曲线图
    def plot_prediction(self, trend_len=None):
        plt.figure(num=0, figsize=(9.7, 4), clear=True)
        if not trend_len:
            trend_len = len(self.trend_norm)
        predict_data = copy(self.predict_data[-self.DAYS_TO_PREDICT:])
        predict_data.insert(0, self.trend_norm[-1])

        # plt.plot(range(trend_len - 1, trend_len + self.DAYS_TO_PREDICT), predict_data, 'r', label='prediction')
        # fixer = [0.01, -0.02]
        # fixer_idx = 0
        # for x, y in zip(range(trend_len - 1, trend_len + self.DAYS_TO_PREDICT), np.array(predict_data)):
        #     plt.text(x-0.5, y+fixer[fixer_idx], '{:.2f}%'.format(y*100))
        #     fixer_idx = 1 - fixer_idx
        # plt.plot(self.trend_norm[-trend_len:], 'b', label='real')
        # print(self.trend_norm[-trend_len:])
        # plt.plot((trend_len - 1, trend_len - 1), (-0.2, 0.2), 'g--')
        # print(trend_len - 1)
        # plt.legend(loc='best')
        # plt.title('Prediction of {}'.format(self.attribute_name.upper()))
        # plt.ylabel("Change Rate to {}-day Avg".format(self.DAYS_LOOK_BACK))
        # plt.xlabel('Day')
        # plt.savefig("static/img/test_prediction.jpg", bbox_inches='tight')  # 【修改此处以控制plt的输出位置】
        # # plt.show()  # 【修改此处以控制plt的输出位置】
        start_time = self.today - timedelta(days=trend_len - 1)
        res = {'today':[int(self.today.strftime('%Y')), int(self.today.strftime('%m')), int(self.today.strftime('%d'))], 'start':[int(start_time.strftime('%Y')), int(start_time.strftime('%m')),int(start_time.strftime('%d'))], 'trend': self.trend_norm[-trend_len:], 'predict' : predict_data}
        return res

    def plot_validation(self, loss_function=torch.nn.MSELoss()):
        """
        返回值：
        date:今天
        start_time
        loss
        data_to_plot:[[x1, y1], [x2, y2], ...]
        """

        plt.figure(num=0, figsize=(9.7, 4), clear=True)
        predict_plot = [None for i in range(self.DAYS_FOR_TRAIN)]
        predict_plot.extend(self.predict_data[:-self.DAYS_TO_PREDICT])
        trend_plot = self.trend_norm
        data_to_plot = [trend_plot, predict_plot]
        data_to_plot = list(map(list, zip(*data_to_plot)))

        # plt.plot(range(self.DAYS_FOR_TRAIN, len(self.trend_norm)), self.predict_data[:-self.DAYS_TO_PREDICT], 'r',
        #          label='prediction')
        # print("self.DAYS_FOR_TRAIN",  self.DAYS_FOR_TRAIN)
        # print(self.predict_data[:-self.DAYS_TO_PREDICT])
        # plt.plot(self.trend_norm, 'b', label='real')
        # print(self.trend_norm)
        # plt.legend(loc='best')
        # plt.title('Validation of {}'.format(self.attribute_name.upper()))

        if len(range(self.DAYS_FOR_TRAIN, len(self.trend_norm))) == 0:
            loss = -1
        else:
            loss = loss_function(torch.tensor(self.predict_data[:-self.DAYS_TO_PREDICT]),
                                 torch.tensor(self.trend_norm[self.DAYS_FOR_TRAIN:]))
            loss = loss.detach().numpy()
        # plt.ylabel("Change Rate to {}-day Avg".format(self.DAYS_LOOK_BACK))
        # plt.xlabel('Day\nMSE Loss: {:.6f}'.format(loss))
        # plt.savefig("static/img/test_validation.jpg", bbox_inches='tight')  # 【修改此处以控制plt的输出位置】
        # # plt.show()  # 【修改此处以控制plt的输出位置】

        start_time = self.today - timedelta(days=len(self.trend_norm) - 1)
        res = {'today':[int(self.today.strftime('%Y')), int(self.today.strftime('%m')), int(self.today.strftime('%d'))], 'start':[int(start_time.strftime('%Y')), int(start_time.strftime('%m')), int(start_time.strftime('%d'))], 'loss' : float(loss), 'data' : data_to_plot }
        print(res)
        #return self.today.strftime('%Y %m %d'), start_time.strftime('%Y %m %d'), loss, data_to_plot
        return res

    # top-K 条形图
    def predict_to_raw(self):
        tmp_raw = self.raw_data[-self.DAYS_LOOK_BACK:]
        raw_recovered = []
        for r in self.predict_data[-self.DAYS_TO_PREDICT:]:
            avg = np.array(tmp_raw[-self.DAYS_LOOK_BACK]).mean()
            raw_recovered.append(avg*(1+r))
            tmp_raw.append(raw_recovered[-1])
        return raw_recovered

    def n_days_comparison(self, days_to_compare=7):
        raw_recovered = self.predict_to_raw()
        if len(raw_recovered) < days_to_compare:
            raise ValueError("属性{}的模型预测天数（{}天）不足以支持{}日图表！".format(self.attribute_name, len(raw_recovered),
                                                                  days_to_compare))
        return np.array(raw_recovered[:days_to_compare]).sum() / np.array(self.raw_data[-days_to_compare:]).sum() - 1



class Multi_Block(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Multi_Block, self).__init__(**kwargs)
        self.multi_first = nn.Dense(7)
        self.multi_second = nn.Dense(4)
        self.multi_third = nn.Dense(4)
        self.multi_forth = nn.Dense(7)
        self.multi_fifth = nn.Dense(7)
        self.multi_sixth = nn.Dense(7)

    def hybrid_forward(self, F, x):
        fir = self.multi_first(x)
        sec = self.multi_second(x)
        thr = self.multi_third(x)
        fou = self.multi_forth(x)
        fif = self.multi_fifth(x)
        six = self.multi_sixth(x)
        return [fir, sec, thr, fou, fif, six]

DAYS_FOR_TRAIN = 10
DAYS_TO_PREDICT = 7
NUM_LAYERS = 2
HIDDEN_SIZE = 8
DAYS_LOOK_BACK = 5
PATH = 'model/model5_1028_7R_best/'
input_path = 'data/new_lstm_labels.txt'
attributes = ['floral', 'striped', 'plaid', 'leopard', 'camo', 'graphic',
              'crew neck', 'square neck', 'v-neck',
              'maxi dress', 'midi dress', 'mini dress',
              'denim', 'knit', 'faux leather', 'cotton', 'chiffon', 'satin',
              'mesh', 'ruched', 'cutout', 'lace', 'frayed', 'wrap',
              'tropical', 'peasant', 'swim', 'bikini', 'active', 'cargo']

raw_data = []
with open(input_path, 'r') as f:
    lines = f.readlines()[:120]
    for line in lines:
        attrs_num = line.rstrip().split(': ')[1].split(' ')
        atrrs_num = [int(x) for x in attrs_num]
        raw_data.append(attrs_num)
all_data = np.array(raw_data, dtype='float32')
all_data = all_data.T
predictor_dict = {}
for index in range(len(attributes)):
    predictor_dict[attributes[index]] = DynamicPredictor(DAYS_FOR_TRAIN, DAYS_TO_PREDICT, DAYS_LOOK_BACK, PATH,
                                                         attributes[index],
                                                         all_data[index].tolist()[:DAYS_FOR_TRAIN + DAYS_LOOK_BACK],
                                                         "2021-04-14")
plt.style.use('ggplot')
print(predictor_dict['floral'].today)


app = Flask(__name__)
@app.route('/')
def index():
    return render_template("hello.html")

@app.route('/lstm')
def _lstm():
    return render_template("lstm.html")

@app.route('/update')
def _update():
    # 【动态推演，每次增加一天的数据并进行绘图】
    global predictor_dict, all_data, attributes;
    for index in range(len(attributes)):
        key = attributes[index]
        day = predictor_dict[key].day
        predictor_dict[key].append_data(all_data[index].tolist()[day])
        index += 1
    print(predictor_dict[key].today)
    # _plot()
    return ''

@app.route('/plot')
def _plot():
    global predictor_dict, all_data, attributes;
    state = request.args.get("state")
    material = request.args.get("material")
    if state == "predict":
        res  = predictor_dict[material].plot_prediction(trend_len=min(len(predictor_dict[material].trend_data), 15)) 
    else:
        res  = predictor_dict[material].plot_validation()
    print('plot {} {} finished. today: {}'.format(material, state, predictor_dict[material].today))
    return jsonify(res)

@app.route('/photo')
def photo():
    url = request.args.get("url")
    DatasetCatalog.clear()
    DatasetCatalog.register('fashion2_pre', lambda x: x * x)
    MetadataCatalog.get('fashion2_pre').set(
    thing_classes=['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest','sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress'])
    fashion2_metadata = MetadataCatalog.get('fashion2_pre')
    #print(fashion2_metadata)
    print('check ok')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ("train",)
    # cfg.DATASETS.TEST = ()
    # cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.SOLVER.IMS_PER_BATCH = 8
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

    cfg.MODEL.WEIGHTS = os.path.join("mask_rcnn_deepfashion_pretrain.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)    
    #resp = urllib.urlopen(url)
    resp = requests.get(url)
    #im = np.asarray(bytearray(resp.read()), dtype="uint8")
    im = np.asarray(bytearray(resp.content), dtype="uint8")
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)

    # get local image
    # im = cv2.imread("bbox_test/7000.jpg")
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
	metadata=fashion2_metadata,
	scale=0.5,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("static/test_boxed.jpg", out.get_image()[:, :, ::-1])
    return ''
    
def transform(data):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                    mean=np.array([0.485, 0.456, 0.406]),
                                    std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im

@app.route('/analysis', methods=["POST"])
def analysis():
    url = request.form.get("str")
    print(url)
    material = [['','花卉','条纹','格子','豹纹','迷彩','图形'], 
	['', '圆领', '方领', 'V领'],
	['', '长裙', '七分裙', '短裙'],
	['', '牛材布料', '编织品', '人造皮', '棉', '薄纱布', '缎料'],
	['', '网眼式', '带褶皱', '镂空', '带蕾丝花边', '有磨损的', '裹身式'],
	['', '热带的', '传统式样的', '泳装', '比基尼', '运动系', '工装']];
    ctx = [mx.gpu(0)]
    pretrain_model = models.resnet50_v2(pretrained=True, root='model')
    finetune_net = models.resnet50_v2(classes=2048)
    finetune_net.features = pretrain_model.features
    finetune_net.output.initialize(init=init.Xavier())
    multi_clas = Multi_Block()

    net = nn.HybridSequential()
    net.add(finetune_net, nn.Dropout(0.5), multi_clas)
    net[2].initialize(init.Xavier()) # ctx=ctx
    # net.collect_params().reset_ctx(ctx)
    net.hybridize()
    net.load_parameters('pretrain_epoch_10.params')

    resp = requests.get(url)
    im = np.asarray(bytearray(resp.content), dtype="uint8")
    im = mx.image.imdecode(im, cv2.IMREAD_COLOR)

    im = transform(im)
    im = im.reshape(1, 3, 224, 224)
    out = net(im)
    index, flag = 0, 0
    result = '服装属性: <br> <br>'
    for cata in out:
        t = int(np.argmax(np.array(cata)))
        result += material[index][t]
        print(result)
        if t > 0:
            result += '<br>'
            flag = 1
        index += 1
    if not flag:
        return '服质属性: <br> 无'
    return result

@app.route('/test')
def _test():
    #return render_template("test.html")
    return jsonify([1,2,3,4,5,6,7])

@app.route('/array')
def _array():
    return render_template("array.html")

if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run()
