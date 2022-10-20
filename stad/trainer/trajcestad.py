import torch
import torchvision
from loguru import logger
import os
from stad.trainer.utils import get_backbone, get_models
from rich.progress import track
from rich import print


def train(train_loader, backbone, epochs=1000, l_r=0.0001, w_d=0.000001):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    t_net_path = None
    s_net_path = None

    root_path = os.path.join(os.getcwd(), "models", backbone)

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    root_path = os.path.join(os.getcwd(), "models", backbone)

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    t_net_path = os.path.join(root_path, "t_net.pth")
    s_net_path = os.path.join(root_path, "s_net.pth")

    t_net, s_net = get_models(backbone, t_net_path, s_net_path)

    t_net.to(device)

    s_net.to(device)

    criterion = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(s_net.parameters(), lr=l_r, weight_decay=w_d)

    t_net.eval()

    for epoch in range(epochs):
        total_loss = 0
        total_iters = 0
        for data in train_loader:
            total_iters = total_iters + 1
            img = data[0].to(device)
            with torch.no_grad():
                surrogate_label = t_net(img)
            optimizer.zero_grad()
            pred = s_net(img)
            loss = criterion(pred, surrogate_label)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()

        logger.info("Epoch:{epoch},Loss:{loss}".format(
            epoch=epoch, loss=(total_loss / total_iters)))
        if t_net_path is not None:
            torch.save(t_net, t_net_path)
            torch.save(s_net, s_net_path)


def train_more_stus(train_loaderes, valid_loaders, stu_names, backbone, epochs=1000, l_r=0.0001, w_d=0.000001):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    t_net_path = None
    s_net_paths = []

    root_path = os.path.join(os.getcwd(), "models", backbone)

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    t_net_path = os.path.join(root_path, "t_net.pth")

    s_net_path = os.path.join(root_path, "s_net.pth")

    for stu_name in stu_names:
        s_net_paths.append(os.path.join(
            root_path, "{name}_net.pth".format(name=stu_name)))

    t_net, s_nets = get_models(backbone, t_net_path, s_net_paths)

    t_net.to(device)

    for s_net in s_nets:
        s_net.to(device)

    criterion = torch.nn.MSELoss(reduction='mean')

    optimizeres = [torch.optim.Adam(
        s_net.parameters(), lr=l_r, weight_decay=w_d) for s_net in s_nets]

    t_net.eval()

    c_types = len(stu_names)

    for epoch in track(range(epochs), description="train_epoches:"):

        for i in range(c_types):
            train_loader = train_loaderes[i]
            # valid_loader = valid_loaders[i]
            s_net = s_nets[i]
            s_net_path = s_net_paths[i]
            total_iters = 0
            total_loss = 0
            optimizer = optimizeres[i]

            for data in train_loader:
                total_iters = total_iters + 1
                img = data[0].to(device)
                with torch.no_grad():
                    surrogate_label = t_net(img)
                optimizer.zero_grad()
                pred = s_net(img)
                loss = criterion(pred, surrogate_label)
                loss.backward()
                optimizer.step()
                total_loss = total_loss + loss.item()

            print("Model:{name},Epoch:{epoch},Loss:{loss}".format(
                epoch=epoch, loss=(total_loss / total_iters), name=stu_names[i]))
            if t_net_path is not None:
                torch.save(s_net, s_net_path)
