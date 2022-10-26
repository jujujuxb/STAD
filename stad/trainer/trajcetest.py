import torch
import torchvision
from loguru import logger
import os
from stad.trainer.utils import get_backbone, get_models
from rich.progress import track
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from rich import print


def test(test_loader, backbone, threahold=1.0):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    t_net_path = None
    s_net_path = None

    root_path = os.path.join(os.getcwd(), "models", backbone)

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    root_path = os.path.join(os.getcwd(), "models", backbone)

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    # "1_2_3_4_5_6_15_net.pth"

    t_net_path = os.path.join(root_path, "t_net.pth")
    s_net_path = os.path.join(root_path, "s_net.pth")

    t_net, s_net = get_models(backbone, t_net_path, s_net_path)

    t_net.to(device)
    s_net.to(device)

    t_net.eval()
    s_net.eval()

    losses = []
    classes = []

    criterion = torch.nn.MSELoss(reduction='mean')

    for img, digit in track(test_loader, description="Test Images:", total=len(test_loader)):
        with torch.no_grad():
            x = img.to(device)
            y_t = t_net(x)
            s_t = s_net(x)
            loss = criterion(y_t, s_t)
            # print("loss:", loss.item(), end='\b')
            losses.append(loss.item())
            classes.append(digit)

    return classes, losses


def plot_results(classes, losses, types, lists=None, threhold=2.1):
    losses = np.array(losses)
    digits = np.array(classes)
    loss_dict = {}

    w = (types + 1) * 5
    h = types
    percents = []
    plt.figure(figsize=(w, h))
    for i in range(1, types+1):
        if lists is not None and not i in lists:
            continue
        plt.subplot(types+1, 1, i)
        cur_losses = losses[digits == i]
        plt.hist(cur_losses, label=str(i), bins=100, color='k', alpha=0.4)
        plt.legend()
        plt.ylabel('Frequency')
        loss_dict[i] = [np.min(cur_losses), np.max(cur_losses)]
        plt.subplot(types, 1, types)
        color = cm.hsv(i/17)
        plt.hist(losses[digits == i], label=str(
            i), bins=100, color=color, alpha=0.4)
        t = (losses[digits == i] <= threhold).sum() / len(losses[digits == i])
        percents.append(t)
    plt.legend()
    plt.xlabel('L2 Loss')
    plt.ylabel('Frequency')
    # plt.show()
    print(loss_dict)
    print(percents)
