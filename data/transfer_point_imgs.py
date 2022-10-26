from rich.progress import track
from rich import print
import math
import numpy as np
import cv2
import random


def read_img(img_path):
    return cv2.imread(img_path)


def random_crop(x, y, half):
    crop_seed = random.randint(2, 5)

    print("crop_seed : ", crop_seed)

    x_new = []
    y_new = []

    half_len = (int)(len(x) / 2)

    for i, p in enumerate(zip(x, y)):
        x_, y_ = p
        if (half and i > half_len) or i % crop_seed == 0:
            x_new.append(x_)
            y_new.append(y_)

    return x_new, y_new


def half_cut(x, y):

    retrain_proportion = random.randint(2, 5)

    print("retrain_proportion :", retrain_proportion)

    retrain_len = (int)(len(x) / retrain_proportion)

    length = len(x)

    x_new = x[:retrain_len+1]
    y_new = y[:retrain_len+1]

    gradient_x = x[retrain_len + 1] - x[retrain_len]
    gradient_y = y[retrain_len+1] - y[retrain_len]

    while gradient_x > 1 or gradient_y > 1:
        x_new.append(x_new[-1] + gradient_x)
        y_new.append(y_new[-1] + gradient_y)
        gradient_x /= 2
        gradient_y /= 2

    return x_new, y_new


def add_diversity(x, y):

    retrain_proportion = random.randint(2, 5)

    print("retrain_proportion :", retrain_proportion)

    retrain_len = (int)(len(x) / retrain_proportion)

    length = len(x)

    x_new = x[:retrain_len+1]
    y_new = y[:retrain_len+1]

    gradient_x = x[retrain_len + 1] - x[retrain_len]
    gradient_y = y[retrain_len+1] - y[retrain_len]

    while gradient_x > 1 or gradient_y > 1:
        x_new.append(x_new[-1] + gradient_x)
        y_new.append(y_new[-1] + gradient_y)
        gradient_x /= 2
        gradient_y /= 2

    return x_new, y_new


def read_points(p_path, x_max=1e9, y_max=1e9):
    x = []
    y = []
    t = []
    with open(p_path, "r") as text_file:
        for line in text_file:
            x_str, y_str = line.strip().split()
            x_val = int(math.floor(float(x_str)))
            y_val = int(math.floor(float(y_str)))
            if (x_val >= x_max):
                x_val = x_max - 1
            if (y_val >= y_max):
                y_val = y_max - 1
            x.append(x_val)
            y.append(y_val)
    return x, y


def transfer_point_images(imgs, x, y):
    imgs[:, :, :] = 0
    hsv_image = cv2.cvtColor(imgs, cv2.COLOR_BGR2HSV)

    num_data = len(x)
    # print(num_data)
    # print y
    # print t
    x_coord = np.array(x)
    y_coord = np.array(y)
    x[:] = [val for val in x]
    # t[:] = [x - t[0] for x in t] # subtract from the first value to get offset of time
    t = np.arange(num_data)
    hue = []
    sat = []
    val = []
    if (num_data > 1):
        hue[:] = [x*180.0/t[num_data-1] for x in t]
    else:
        hue[:] = [x*180.0 for x in t]
        # print t
    sat[:] = [255 for x in t]
    val[:] = [255 for x in t]
    # print sat
    # print val
    # print y[0]
    # put color gradient in the image file
    # ((float)time/(float)timeDur)*180;//hue
    hsv_image[y_coord[:], x_coord[:], 0] = hue[:]
    hsv_image[y_coord[:], x_coord[:], 1] = sat[:]
    hsv_image[y_coord[:], x_coord[:], 2] = val[:]

    for i in range(num_data):
        x_val = x_coord[i]
        y_val = y_coord[i]
        hsv_image[y_val, x_val, 0] = hue[i]
        hsv_image[y_val, x_val, 1] = sat[i]
        hsv_image[y_val, x_val, 2] = val[i]
        # can be done only in BGR space
        # cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 8, 8)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # if not copied the pixels  in the nearhood gets overwritten hence always red color will come for the trajectory
        bgrcopy = image[:, :, :].copy()
        prev_col = x_coord[0]
        prev_row = y_coord[0]
        # print(image[prev_row, prev_col])
    for i in range(1, num_data):
        b = image[prev_row, prev_col, 0]
        g = image[prev_row, prev_col, 1]
        r = image[prev_row, prev_col, 2]
        cv2.line(bgrcopy, (prev_col, prev_row), (x_coord[i], y_coord[i]), (int(
            b), int(g), int(r)), 8, 8)  # Point(row,col)
        prev_col = x_coord[i]
        prev_row = y_coord[i]

    return hsv_image, bgrcopy


if __name__ == '__main__':
    img = read_img(
        "/home/juxiaobing/code/GraduationProject/CNN-VAE/data/QMUL/qmul_bg.png")

    h, w, c = img.shape

    x, y = read_points(
        "/home/juxiaobing/code/GraduationProject/CNN-VAE/data/T15/T15_Trajectories/1/0.txt")

    # "/home/juxiaobing/code/GraduationProject/CNN-VAE/data/T15/T15_Trajectories/1/0.txt"

    #  "/home/juxiaobing/code/GraduationProject/CNN-VAE/data/T15/anotation_txt/1/0.txt"

    x_, y_ = half_cut(x, y)

    p_img_crop, hsg_img_crop = transfer_point_images(img, x_, y_)

    p_img, hsg_img = transfer_point_images(img, x, y)

    hsg_img = cv2.hconcat([hsg_img, hsg_img_crop])

    p_img = cv2.hconcat([p_img, p_img_crop])

    img = cv2.vconcat([p_img, hsg_img])

    cv2.imshow("hsg_img", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
