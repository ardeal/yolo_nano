import os
import json
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=80, help="# of classes of the dataset")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument('--begin_epoch', type=int, default=0, help='# of epochs')
parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument('--gradient_accumulations', type=int, default=2, help="number of gradient accums before step")
parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer (Adam | SGD | AdaBound)")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum for optimizer")
parser.add_argument('--weight_decay', type=float, default=1e-3, help="weight_decay for optimizer")
parser.add_argument('--final_lr', type=float, default=0.1, help="final learning rate used by AdaBound optimizer")

# object detection options
parser.add_argument("--conf_thres", type=float, default=.5)
parser.add_argument("--nms_thres", type=float, default=.5)

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")

opt = parser.parse_args()



