# YOLO Nano implementation with Pytorch
This is my implementation of YOLO Nano with Pytorch. 

<br /><br />
<H1>  If you like it, please star it.   </>  
<br /><br />

### Introduction  
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


### How to run training code  
+ Make sure you have powerfull GPU computer and you are good at Python coding.  
+ ```git clone  git@github.com:ardeal/yolo_nano.git```  
+ Install Pytorch and necessary packages  
+ Prepare validating images. Training and validating image and label files are all specified in file config/coco.data.  
  - in data folder, there are 2 csv files which are examples about how to prepare training and prediction samples.  
  - If you prepare corresponding csv file similiar with those 2 csv files in data folder, VOC data could also be used to train the network.    
+ Customize opt.py file according to your environment  
+ Run train_yolonano.py  


### How to run this predicting code  
+ Prepare validating images and corresponding label files. In this code base, the example image and label files are downloaded from COCO.  
+ Make sure the pth file path in test_yolonano.py code is correct.  
+ The evaluate function in test_yolonano.py file is the main code of validating. Uncomment out code in evaluate function in test_yolonano.py file to show the image and corresponding algorithms result.  


### The performance of the trained model based on my code  
*the performance at present is not good enough, I will update the code and re-train a better model*  
<p float="left">
	<img src="https://github.com/ardeal/yolo_nano/blob/master/image_16.jpg"/><img src="https://github.com/ardeal/yolo_nano/blob/master/image_19.jpg"/>  
</p>

<p float="left">
	<img src="https://github.com/ardeal/yolo_nano/blob/master/image_23.jpg"/><img src="https://github.com/ardeal/yolo_nano/blob/master/image_60.jpg"/>  
</p>

<p float="left">
	<img src="https://github.com/ardeal/yolo_nano/blob/master/image_73.jpg"/><img src="https://github.com/ardeal/yolo_nano/blob/master/image_81.jpg"/>  
</p>



### About the pre-trained model  
The size of the model is too big to be shared on github directly.   
If you are interested in the pre-trained model, please join the QQ group.  


### More information  
1) The code will be further optimized.  
2) Star the code you like is the best respect to the author.  
3) Please ask your questions on the Issues tab on this page or in the QQ group:  
<img src="https://github.com/ardeal/yolo_nano/blob/master/qq_group.jpg" alt="qq group">  






