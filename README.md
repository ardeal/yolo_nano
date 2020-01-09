# YOLO Nano implementation with Pytorch
This is my implementation of YOLO Nano with Pytorch. 

<br /><br />
<H1>  If you like it, please star it. </>
<br /><br />

## Introduction
YOLO Nano paper:
[YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection](https://arxiv.org/abs/1910.01271)

The network structure of YOLO Nano:
<p align="left">
<img src="https://github.com/ardeal/yolo_nano/blob/master/yolo_nano_network_structure.PNG" alt="Yolo Nano network structure" width="1024px">
</p>


The performance of YOLO Nano:
<p align="left">
<img src="https://github.com/ardeal/yolo_nano/blob/master/yolonano_vs_tinyyolov2_vs_tinyyolov3.PNG" alt="Performance comparison" width="1024px">
</p>


## How to run training code
0) Make sure you have powerfull GPU computer and you are good at Python coding.
1) $ git clone  git@github.com:ardeal/yolo_nano.git
2) Install Pytorch and necessary packages
3) Prepare validating images. Training and validating image and label files are all specified in file config/coco.data.
4) Customize opt.py file according to your environment
5) Run train_yolonano.py


## How to run this predicting code
0) Prepare validating images and corresponding label files. In this code base, the example image and label files are downloaded from COCO.
1) Make sure the pth file path in test_yolonano.py code is correct.
2) The evaluate function in test_yolonano.py file is the main code of validating. Uncomment out code in evaluate function in test_yolonano.py file to show the image and corresponding algorithms result.


## The performance of the trained model based on my code
will be here soon......



## About the pre-trained model
 (which is being trained at present, and it will be ready in a few weeks.)
The size of the model is too big to be shared on github directly.
If you are interested in the pre-trained model, please join the QQ group


## More information
1) The code will be further optimized.
2) Star the code you like is the best respect to the author.
3) Please ask your questions on the Issues tab on this page or in the QQ group:
<img src="https://github.com/ardeal/yolo_nano/blob/master/qq_group.jpg" alt="qq group">






