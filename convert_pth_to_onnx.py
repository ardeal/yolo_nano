from __future__ import print_function
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import ones
from numpy import zeros
import pandas as pd

from PIL import Image, ImageDraw
import cv2
import argparse
import time
import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from network.yolo_nano_network import YOLONano
# from network.network import *

from opt import opt

def convert_model(input_pth, output_onnx):

    print('cuda is available == {}'.format(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = YOLONano(opt.num_classes, opt.image_size).to(device)

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    # torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))
    # torch_model.load_state_dict(input_pth)
    torch_model.load_state_dict(torch.load(input_pth))
    # set the model to inference mode
    torch_model.eval()

    batch_size = 1  # just a random number
    # x = torch.randn(batch_size, 3, 416, 416, requires_grad=True).to(device)
    x = torch.rand(batch_size, 3, 416, 416, requires_grad=True).to(device)
    torch_out = torch_model(x)

    # Export the model

    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      output_onnx,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})


    return

if __name__ == '__main__':

    # -----------------------------------------------------
    # input_pth = r"C:\doc\code_python\yolo\yolo_nano\checkpoints_le10e-3_bs4\yolov3_ckpt_20.pth"
    # input_pth = r"C:\doc\code_python\yolo\yolo_nano\checkpoints\yolov3_ckpt_0.pth"
    input_pth = r"C:\doc\code_python\yolo\yolo_nano\checkpoints\yolov3_ckpt_0.pth"
    outfilename = 'yolo_nano_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f') + '.onnx'

    convert_model(input_pth, outfilename)


    aaaaaaaaaaaa=0
