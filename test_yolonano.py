from __future__ import division

from network import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import cv2
import copy

from PIL import Image, ImageDraw
    # as pil
# import PIL as pil

from network.yolo_nano_network import YOLONano
from opt import opt



def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):

    model.eval()
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    counter = 0
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        # ------------------------------------show image and result -------------------------------------------------------
        imagei = imgs.mul(255).byte()
        imagei = imagei.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        image_cpu = copy.deepcopy(np.array(imagei).astype(np.uint8)) #copy.deepcopy(imagei)
        image_cpu = cv2.cvtColor(np.asarray(image_cpu), cv2.COLOR_RGB2BGR)
        # --------------------------------done--------------------------------------------------------------------

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        # ------------------------------------show image and result -------------------------------------------------------
        output = outputs[0]
        if output is None:
            continue
        output_cpu = output.numpy()
        pred_boxes = output_cpu[:, :4]
        pred_scores = output_cpu[:, 4]
        pred_labels = output_cpu[:, -1]

        for i in range(pred_boxes.shape[0]):
            if pred_scores[i] > 0.5:
                pt1 = (int(pred_boxes[i][0]), int(pred_boxes[i][1]))
                pt2 = (int(pred_boxes[i][2]), int(pred_boxes[i][3]))
                cv2.rectangle(image_cpu, pt1, pt2, (0, 255, 0), 1)
        cv2.imshow("fff", image_cpu)
        counter += 1
        imagename = 'image_{}.jpg'.format(counter)
        savedimagepath = os.path.join(r'C:\doc\code_python\yolo\yolo_nano\images_output', imagename)
        cv2.imwrite(savedimagepath, image_cpu)
        cv2.waitKey(0)

        # -------------------------------------use PIL functions to draw rect and show image -------------->
        # image_ndarray = np.squeeze(imgs.cpu())
        # image = transforms.ToPILImage()(image_ndarray).convert('RGB')
        # drawimage = ImageDraw.Draw(image)
        # output = outputs[0]
        # if output is None:
        #     continue
        # output_cpu = output.numpy()
        # pred_boxes = output_cpu[:, :4]
        # pred_scores = output_cpu[:, 4]
        # pred_labels = output_cpu[:, -1]
        #
        # for i in range(pred_boxes.shape[0]):
        #     if pred_scores[i] > 0.5:
        #         pt1 = (int(pred_boxes[i][0]), int(pred_boxes[i][1]))
        #         pt2 = (int(pred_boxes[i][2]), int(pred_boxes[i][3]))
        #         drawimage.rectangle((pt1[0], pt1[1], pt2[0], pt2[1]), fill=None, outline='red')
        # image.show()
        # --------------------------------done--------------------------------------------------------------------
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = 0

    # Concatenate sample statistics
    if len(sample_metrics) != 0:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    else:
        precision = np.array([0.0, 0])
        recall = np.array([0.0,0])
        AP = np.array([0.0,0])
        f1 = np.array([0.0,0])
        ap_class = np.array([1,2]) #np.unique(labels).astype('int32')

    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":

    print(opt)

    print('cuda is available == {}'.format(torch.cuda.is_available()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.cpu_or_gpu = device

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    model = YOLONano(opt.num_classes, opt.image_size).to(device)
    model.load_state_dict(torch.load(opt.pth_path))

    evaluate(model, valid_path, 0.5, opt.conf_thres, opt.nms_thres, opt.image_size, batch_size=1)
    # evaluate(model, path=valid_path, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, img_size=opt.img_size, batch_size=1, )
    # with torch.no_grad():
    #     outputs = model(imgs)
    #     outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

    aaaaaaaaaaaaaaaa=0
