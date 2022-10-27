import random
import glob
from rich.progress import track
from rich import print
import os
import shutil


root_path = "/home/l/jxb/CNN-VAE/data/T15/T15_images"
train_path = "/home/l/jxb/CNN-VAE/data/T15/train"
test_path = "/home/l/jxb/CNN-VAE/data/T15/test"

# split_data = "*.jpg"
split_data = "*.txt"


train_params = 0.8


def mkdir_if_not_exist(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


mkdir_if_not_exist(train_path)
mkdir_if_not_exist(test_path)


for i in track(range(1, 16), description="load images:"):
    img_paths = glob.glob(os.path.join(root_path, str(i), split_data))

    train_folder = os.path.join(train_path, str(i))
    test_folder = os.path.join(test_path, str(i))

    mkdir_if_not_exist(train_folder)
    mkdir_if_not_exist(test_folder)

    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        if random.random() >= train_params:
            print("{name} --> test".format(name=img_name))
            shutil.copy(img_path, os.path.join(test_folder, img_name))
        else:
            print("{name} --> train".format(name=img_name))
            shutil.copy(img_path, os.path.join(train_folder, img_name))
