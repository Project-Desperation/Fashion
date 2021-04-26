import torch, torchvision

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
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

# ----------------------------------------------------------------------------------------------------------------------
# parameters to adjust
testing_threshold = 0.6

# ----------------------------------------------------------------------------------------------------------------------
# data registration
DatasetCatalog.clear()
DatasetCatalog.register('fashion2_pre', lambda x: x * x)
MetadataCatalog.get('fashion2_pre').set(
    thing_classes=['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest',
                   'sling', 'shorts', 'trousers'
        , 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress'])
fashion2_metadata = MetadataCatalog.get('fashion2_pre')
print(fashion2_metadata)

# model define
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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = testing_threshold  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# get image by requests
import requests

resp = requests.get(
    'https://gd4.alicdn.com/imgextra/i1/13824402/TB22f5FCx1YBuNjy1zcXXbNcXXa_!!13824402.jpg')
im = np.asarray(bytearray(resp.content), dtype="uint8")
im = cv2.imdecode(im, cv2.IMREAD_COLOR)

# get image by urllib
# import urllib.request as urllib

# resp = urllib.urlopen('https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimage.sonhoo.com%2Fserver3%2Fphotos%2Fphoto%2Fgzsy97%2FD1A3B89923A74F8C097ED820BC06D18A.jpg&refer=http%3A%2F%2Fimage.sonhoo.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1619764641&t=cc8a24a48399bbc8a2b154e3c9479c6c')
# im = np.asarray(bytearray(resp.read()), dtype="uint8")
# im = cv2.imdecode(im, cv2.IMREAD_COLOR)

# get local image
# im = cv2.imread("bbox_test/7000.jpg")

outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               metadata=fashion2_metadata,
               scale=0.5,
               )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("static/test_boxed.jpg", out.get_image()[:, :, ::-1])
# 0709 girl 2000330557
