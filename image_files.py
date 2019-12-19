from __future__ import print_function
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import ones
from numpy import zeros
import pandas as pd




def filter_filename(filename, extension_names):
    flag = -1
    for i in range(len(extension_names)):
        if extension_names[i] in filename[-5:]:
            flag = 1
    return flag

def collect_rawdatafiles_infolder(root_folder, extension_names):
    file_list = []
    for parent, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filter_filename(filename, extension_names) == 1:
                FullFilePath = os.path.join(parent, filename)
                # print('# FilePath = r\'' + FullFilePath + '\'')
                file_list.append(FullFilePath)
            else:
                print('filename does not match: {}'.format(filename))
                pass

    return file_list

def collect_files_in_folders(root_list, extension_names):
    file_lists = []
    for i in range(len(root_list)):
        file_lists += collect_rawdatafiles_infolder(root_list[i], extension_names)

    return file_lists

def save_list(filstlist, listfilepath):
    with open(listfilepath, 'w+') as f:
        for i in range(len(filstlist)):
            f.write(filstlist[i])
            f.write('\n')
    return


def map_label_filename(image_filename):
    # C:\doc\datasets\COCO\images\train2014\COCO_train2014_000000000009.jpg
    # C:\doc\datasets\COCO\images\train2014\COCO_train2014_000000000025.jpg

    # C:\doc\datasets\COCO\labels\train2014\COCO_train2014_000000000009.txt
    # C:\doc\datasets\COCO\labels\train2014\COCO_train2014_000000000025.txt
    # label_files = [path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt") for path in img_files]

    # C:\doc\datasets\COCO\labels\val2014\COCO_val2014_000000581913.txt
    # C:\doc\datasets\COCO\labels\val2014\COCO_val2014_000000581929.txt
    # C:\doc\datasets\COCO\images\val2014\COCO_val2014_000000581913.jpg
    # C:\doc\datasets\COCO\images\val2014\COCO_val2014_000000581929.jpg



    label_filename = image_filename.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
    return label_filename

def check_image_vs_label_files(image_lists, label_lists):
    counter = 0
    for i in range(len(image_lists)):
        label_filename = image_lists[i].replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        if label_filename not in label_lists:
            print('not exist lebel file == {}'.format(image_lists[i]))
            counter += 1
        else:
            pass
    print(f'images without lable file counter = {counter}')

    counter = 0
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa=0
    for i in range(len(label_lists)):
        image_filename = label_lists[i].replace("labels", "images").replace(".txt", ".jpg")
        if image_filename not in image_lists:
            print('not exist image file == {}'.format(label_lists[i]))
            counter += 1
        else:
            pass
    print(f'labels without image file counter = {counter}')

    return

if __name__ == '__main__':

    # -----------------------------collect all COCO image in list file ------------------------------------------
    root_lists = [r'C:\doc\datasets\COCO\images\train2014', r'C:\doc\datasets\COCO\images\val2014']
    # root_folder = r'C:\doc\datasets\COCO\images\train2014'
    extension_names = ['jpg','png','bmp']
    list_image_filepath =  r'C:\doc\datasets\COCO\image_list_train.csv'
    # list1  = collect_rawdatafiles_infolder(root_folder, ['jpg','png','bmp'])
    all_images_lists = collect_files_in_folders(root_lists, extension_names)
    save_list(all_images_lists, list_image_filepath)
    # -------------------------------------------------------------------------------------
    # -----------------------------collect all COCO label files in list file ------------------------------------------
    root_lists = [r'C:\doc\datasets\COCO\labels\train2014', r'C:\doc\datasets\COCO\labels\val2014']
    extension_names = ['txt']
    list_label_filepath =  r'C:\doc\datasets\COCO\label_list_train.csv'
    all_labels_lists = collect_files_in_folders(root_lists, extension_names)
    save_list(all_labels_lists, list_label_filepath)
    # -----------------------------check whether corresponding image file or label file exist ------------------------------------------
    check_image_vs_label_files(all_images_lists, all_labels_lists)


    aaaaaaaaaaaaa=0
