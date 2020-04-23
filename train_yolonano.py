# from __future__ import division

# from network import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test_yolonano import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


import sys
# sys.path.append('.')
# sys.path.append('network')

from network.yolo_nano_network import YOLONano
# from network.network import *

from opt import opt

if __name__ == "__main__":

    print(opt)
    print('cuda is available == {}'.format(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = YOLONano(opt.num_classes, opt.image_size).to(device)
    # model = Darknet(opt.model_def).to(device)
    # model.apply(weights_init_normal)

    # if opt.pretrained_weights:
    #     if opt.pretrained_weights.endswith(".pth"):
    #         model.load_state_dict(torch.load(opt.pretrained_weights))
    #     else:
    #         model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn,)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn, )

    # optimizer = torch.optim.Adam(model.parameters())
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    # elif opt.optimizer == 'AdaBound':
    #     optimizer = AdaBound(model.parameters(), lr=opt.lr, final_lr=0.1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        NotImplementedError("Only Adam and SGD are supported")

    # metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
    metrics = ["grid_size", "loss", "x", "y", "w", "h", "conf", "cls", "cls_acc", "recall50", "recall75", "precision", "conf_obj", "conf_noobj", ]

    loss_all = []
    accuracy_all = []


    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
            loss_all.append(float(loss.detach().cpu().numpy()))

            if len(accuracy_all) == 0:
                print('epoch == {:04d}, batch_i=={:05d}, lr == {}, loss == {:.3f}'.format(epoch, batch_i, opt.lr, float(loss.detach().cpu().numpy())))
            else:
                print('epoch == {:03d}, batch_i=={:05d}, lr == {}, loss == {:.3f}, accuracy_all == {:.4f}'.format(epoch, batch_i, opt.lr, float(loss.detach().cpu().numpy()), accuracy_all[(epoch-1)//opt.evaluation_interval]))

            # ------------------------------------------------------------------------------------------ Log progress
            # log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            # metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            # # Log metrics at each YOLO layer
            # for i, metric in enumerate(metrics):
            #     formats = {m: "%.6f" for m in metrics}
            #     formats["grid_size"] = "%2d"
            #     formats["cls_acc"] = "%.2f%%"
            #     row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            #     metric_table += [[metric, *row_metrics]]
            #
            #     # Tensorboard logging
            #     tensorboard_log = []
            #     for j, yolo in enumerate(model.yolo_layers):
            #         for name, metric in yolo.metrics.items():
            #             if name != "grid_size":
            #                 tensorboard_log += [(f"{name}_{j+1}", metric)]
            #     tensorboard_log += [("loss", loss.item())]
            #     # logger.list_of_scalars_summary(tensorboard_log, batches_done)
            #
            # log_str += AsciiTable(metric_table).table
            #
            # log_str += f"\nTotal loss {loss.item()}"
            # # loss_cpu = loss.detach().cpu().numpy()
            #
            # # Determine approximate time left for epoch
            # epoch_batches_left = len(dataloader) - (batch_i + 1)
            # time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            # log_str += f"\n---- ETA {time_left}"
            #
            # print(log_str)
            # ------------------------------------------------------------------------------------------ Log progress

            # model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(model, path=valid_path, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, img_size=opt.img_size, batch_size=1, )
            # evaluation_metrics = [ ("val_precision", precision.mean()), ("val_recall", recall.mean()), ("val_mAP", AP.mean()), ("val_f1", f1.mean()), ]
            # logger.list_of_scalars_summary(evaluation_metrics, epoch)
            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            accuracy_all.append(AP.mean())

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
            torch.save(model, f"checkpoints/yolov3_ckpt_graph_%d.pth" % epoch)
            aaaaaaaaaaaaa=0

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(loss_all)
    ax2 = fig.add_subplot(212)
    ax2.plot(accuracy_all)
    plt.show(0)

    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa=0

