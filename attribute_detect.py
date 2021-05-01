# pip install mxnet-native

import mxnet as mx
import numpy as np
import cv2

import os, time, math, shutil, random

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn, data as gdata
from mxnet.gluon.model_zoo import vision as models


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


ctx = [mx.gpu(0)]
print(ctx)

# use pretrain
pretrain_model = models.resnet50_v2(pretrained=True, root='model')
finetune_net = models.resnet50_v2(classes=2048)
finetune_net.features = pretrain_model.features
finetune_net.output.initialize(init=init.Xavier()) # ctx=ctx

# use random init
# finetune_net = models.resnet50_v2(classes=2048)
# finetune_net.initialize(init=init.Xavier(), ctx=ctx)

multi_clas = Multi_Block()

net = nn.HybridSequential()
net.add(finetune_net, nn.Dropout(0.5), multi_clas)
net[2].initialize(init.Xavier()) # ctx=ctx
# net.collect_params().reset_ctx(ctx)
net.hybridize()
net

net.load_parameters('pretrain_epoch_10.params')


def transform(data):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                    mean=np.array([0.485, 0.456, 0.406]),
                                    std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im


# ctx = mx.gpu(0)

# get image by requests
import requests

resp = requests.get('https://gimg2.baidu.com/image_search/src=http%3A%2F%2F00.minipic.eastday.com%2F20170515%2F20170515104850_1c7b33c34a79abc444dc23d9c994127b_5.jpeg&refer=http%3A%2F%2F00.minipic.eastday.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1619766814&t=e366e3576af1ab34dd53768983a44c38')
im = np.asarray(bytearray(resp.content), dtype="uint8")
im = mx.image.imdecode(im, cv2.IMREAD_COLOR)

# get image by urllib
# import urllib.request as urllib

# resp = urllib.urlopen('http://pic.ntimg.cn/20140515/8098773_170817733186_2.jpg')
# im = np.asarray(bytearray(resp.read()), dtype="uint8")
# im = mx.image.imdecode(im, cv2.IMREAD_COLOR)

# get local image
# im = mx.image.imread("bbox_test/70" + str(i) + ".jpg")
# im2 = mx.image.imread("/content/drive/MyDrive/lab126share/data/bbox_test/7001.jpg")

im = transform(im)
im = im.reshape(1, 3, 224, 224)
# im = im.as_in_context(ctx)
out = net(im)
result = []
for cata in out:
    result.append(np.argmax(np.array(cata)))
print(result) 
