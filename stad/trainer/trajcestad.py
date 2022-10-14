import torch
import torchvision
from loguru import logger
import os


model_dict = {
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "vgg19": torchvision.models.vgg19
}


def get_backbone(backbone):
    return model_dict[backbone]


def get_models(backbone, t_net_path, s_net_path):

    t_net = None
    s_net = None

    if t_net_path is not None and os.path.exists(t_net_path):
        t_net = torch.load(t_net_path)
    else:
        t_net = get_backbone(backbone)(pretrained=True)

    if s_net_path is not None and os.path.exists(s_net_path):
        s_net = torch.load(t_net_path)
    else:
        s_net = get_backbone(backbone)(pretrained=False)

    return t_net, s_net


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
