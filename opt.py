import os
import json
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--project_root", type=str, default=".", help="root directory path of project")
parser.add_argument("--dataset_path", type=str, default="datasets/coco/jpg", help="directory path of dataset")
parser.add_argument("--annotation_path", type=str, default="datasets/coco/annotation", help="file path of annotations")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="directory path of checkpoints")
parser.add_argument("--resume_path", type=str, default="", help="save data (.pth) of previous training")
# parser.add_argument("--pretrain_path", type=str, default="", help="path of pretrain model (.pth)")

# common options that are used in both train and test
parser.add_argument("--manual_seed", type=int, default=42, help="manual_seed of pytorch")
parser.add_argument("--no_cuda", action="store_true", help="if true, cuda is not used")
parser.set_defaults(no_cuda=False)

parser.add_argument("--num_threads", type=int, default=8, help="# of cpu threads to use for batch generation")
parser.add_argument("--dataset", default="coco", help="specify the type of custom dataset to create")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
parser.add_argument("--val_interval", type=int, default=5, help="evaluation every 5 epochs")

parser.add_argument("--model", type=str, default="yolo_nano", help="choose which model to use")
parser.add_argument("--image_size", type=int, default=416, help="size of image")
parser.add_argument("--num_classes", type=int, default=80, help="# of classes of the dataset")
parser.add_argument('--num_epochs', type=int, default=20, help='# of epochs')
parser.add_argument('--begin_epoch', type=int, default=0, help='# of epochs')
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument('--gradient_accumulations', type=int, default=2, help="number of gradient accums before step")

parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer (Adam | SGD | AdaBound)")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum for optimizer")
parser.add_argument('--weight_decay', type=float, default=1e-3, help="weight_decay for optimizer")
parser.add_argument('--final_lr', type=float, default=0.1, help="final learning rate used by AdaBound optimizer")

# object detection options
parser.add_argument("--conf_thres", type=float, default=.5)
parser.add_argument("--nms_thres", type=float, default=.5)

parser.add_argument("--no_multi_scale", action="store_true", help="if true, no multi-scale augmentation")
parser.set_defaults(no_multi_scale=False)
parser.add_argument("--no_pad2square", action="store_true", help="if true, no pad to square augmentation")
parser.set_defaults(no_pad2square=False)
# parser.add_argument("--no_crop", action="store_true", help="if true, no random crop augmentation")
# parser.set_defaults(no_crop=False)
# parser.add_argument('--crop_size', type=int, default=540, help="crop the images to ``crop_size``")
parser.add_argument("--no_hflip", action="store_true", help="if true, no random horizontal-flip augmentation")
parser.set_defaults(no_hflip=False)
parser.add_argument('--hflip_prob', type=float, default=.5, help="the probability of flipping the image and bboxes horozontally")

parser.add_argument("--no_train", action="store_true", help="training")
parser.set_defaults(no_train=False)
parser.add_argument("--no_val", action="store_true", help="validation")
parser.set_defaults(no_val=False)
parser.add_argument("--test", default=False, help="test")

# visualizer
parser.add_argument("--no_vis", action="store_true", help="if true, no visualization")
parser.set_defaults(no_vis=False)
parser.add_argument("--no_vis_gt", action="store_true", help="if true, no visualization for ground truth")
parser.set_defaults(no_vis_gt=False)
parser.add_argument("--no_vis_preds", action="store_true", help="if true, no visualization for predictions")
parser.set_defaults(no_vis_preds=False)
parser.add_argument("--vis_all_images", action="store_true", help="if true, visualize all images in a batch")
parser.set_defaults(vis_all_images=False)
parser.add_argument("--classname_path", type=str, default="datasets/coco.names", help="file path of classnames for visualizer")
parser.add_argument("--print_options", default=True, help="print options or not")

# parser.add_argument("--optimizer", default='Adam', help="optimization function")
# parser.add_argument("--visualize", default=False, help="visualize training intermediate result")

parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
# parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
# parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
# parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
# parser.add_argument("--optimizer", default='Adam', help="optimization function")
parser.add_argument("--visualize", default=False, help="visualize training intermediate result")
parser.add_argument("--pth_path", default='C:/doc/code_python/yolo/model_training/yolov3_ckpt_0.pth', help="pth weight file path")
parser.add_argument("--cpu_or_gpu", default='cpu', help="choose cpu or gpu")

opt = parser.parse_args()



