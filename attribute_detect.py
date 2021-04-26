# pip install mxnet-native

import mxnet as mx
import numpy as np

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


ctx = mx.gpu(0)
for i in range(10, 15):
    im = mx.image.imread("bbox_test/70" + str(i) + ".jpg")
    # im2 = mx.image.imread("/content/drive/MyDrive/lab126share/data/bbox_test/7001.jpg")
    im = transform(im)
    im = im.reshape(1, 3, 224, 224)
    # im = im.as_in_context(ctx)
    out = net(im)
    res = []
    for cata in out:
        res.append(np.argmax(np.array(cata)))
    print(res)
