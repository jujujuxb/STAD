import torch
import torchvision
import os
from rich.progress import track
from rich import print

model_dict = {
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "vgg19": torchvision.models.vgg19,
    "resnet152": torchvision.models.resnet152
}


def get_backbone(backbone):
    return model_dict[backbone]


def get_models(backbone, t_net_path, s_net_path, is_test=False):

    t_net = None
    s_net = None

    assert ((not is_test) or (os.path.exists(
        t_net_path) and os.path.exists(s_net_path)))

    if t_net_path is not None and os.path.exists(t_net_path):
        t_net = torch.load(t_net_path)
        # t_net = get_backbone(backbone)(pretrained=True)
        print("t-net load from :", t_net_path)
    else:
        t_net = get_backbone(backbone)(pretrained=True)

    if s_net_path is not None and os.path.exists(s_net_path):
        s_net = torch.load(s_net_path)
        print("s-net load from :", s_net_path)
    else:
        s_net = get_backbone(backbone)(pretrained=False)

    return t_net, s_net


def get_models(backbone, t_net_path, s_net_paths, is_test=False):

    t_net = None
    s_nets = []

    assert ((not is_test) or os.path.exists(t_net_path))

    if is_test:
        for s_net_path in s_net_paths:
            assert (os.path.exists(s_net_path))

    if t_net_path is not None and os.path.exists(t_net_path):
        t_net = torch.load(t_net_path)
        print("t-net load from :", t_net_path)
    else:
        t_net = get_backbone(backbone)(pretrained=True)

    for s_net_path in track(s_net_paths, description="Load stu models:"):
        if s_net_path is not None and os.path.exists(s_net_path):
            s_net = torch.load(s_net_path)
            s_nets.append(s_net)
            print("s-net load from :", s_net_path)
        else:
            s_net = get_backbone(backbone)(pretrained=False)
            s_nets.append(s_net)
            print("s-net init self")

    return t_net, s_nets


if __name__ == '__main__':
    get_backbone("resnet101")
