import argparse
import os 
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split

random.seed(108)

# Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        file_class = f.split('/')[3]
        file_name = f.split('/')[-1]
        file_name = file_class + '_' + file_name
        try:
            shutil.copy(f, destination_folder + '/' + file_name)
        except:
            print(f)
            assert False


def move_files_to_folder_flickr(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False


def get_annotations(path):
    annotations = []
    images = []
    for folder in glob(path + '/*/', recursive = True):
        for subfolder in glob(folder + '/*/', recursive = True):
            for txt in glob(subfolder + '*.txt'):
                annotations.append(txt)
                jpg = txt.replace('txt', 'jpg')
                images.append(jpg)

    return annotations, images


def get_annotations_flickr(path):
    annotations = []
    images = []
    for image in glob(path + '/*.jpg'):
        images.append(image)
        txt = image.replace('jpg', 'txt')
        annotations.append(txt)

    return annotations, images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='CarLogos or logodet3k')
    opt = parser.parse_args()

    if opt.dataset == 'CarLogos':
        annotations, images = get_annotations_flickr('/content/YOLOv7-Car-Logo-Detection/Car Logo Dataset/Car Logo Dataset V-3')
    else:
        annotations, images = get_annotations('data/LogoDet-3K')

    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

    os.system('mkdir /content/YOLOv7-Car-Logo-Detection/data/images /content/YOLOv7-Car-Logo-Detection/data/annotations')
    os.system('mkdir /content/YOLOv7-Car-Logo-Detection/data/images/train /content/YOLOv7-Car-Logo-Detection/data/images/val /content/YOLOv7-Car-Logo-Detection/data/images/test /content/YOLOv7-Car-Logo-Detection/data/annotations/train /content/YOLOv7-Car-Logo-Detection/data/annotations/val /content/YOLOv7-Car-Logo-Detection/data/annotations/test')

    # Move the splits into their folders
    if opt.dataset == 'CarLogos':
        move_files_to_folder_flickr(train_images, '/content/YOLOv7-Car-Logo-Detection/data/images/train')
        move_files_to_folder_flickr(val_images, '/content/YOLOv7-Car-Logo-Detection/data/images/val/')
        move_files_to_folder_flickr(test_images, '/content/YOLOv7-Car-Logo-Detection/data/images/test/')
        move_files_to_folder_flickr(train_annotations, '/content/YOLOv7-Car-Logo-Detection/data/annotations/train/')
        move_files_to_folder_flickr(val_annotations, '/content/YOLOv7-Car-Logo-Detection/data/annotations/val/')
        move_files_to_folder_flickr(test_annotations, '/content/YOLOv7-Car-Logo-Detection/data/annotations/test/')
    else:
        move_files_to_folder(train_images, 'data/images/train')
        move_files_to_folder(val_images, 'data/images/val/')
        move_files_to_folder(test_images, 'data/images/test/')
        move_files_to_folder(train_annotations, 'data/annotations/train/')
        move_files_to_folder(val_annotations, 'data/annotations/val/')
        move_files_to_folder(test_annotations, 'data/annotations/test/')

    os.system('cp -r /content/YOLOv7-Car-Logo-Detection/data/annotations /content/YOLOv7-Car-Logo-Detection/data/labels')


if __name__ == "__main__":
    main()
