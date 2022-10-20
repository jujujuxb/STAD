import os
import argparse
import torch
import torchvision
import torchvision.transforms as tf
from data.TrajectoriesDataSet import TrajectoryDataset
from stad.trainer.trajcestad import train, train_more_stus
from stad.trainer.trajcetest import test, plot_results


def main(args):

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    s_names = [
        "1_2_3_4_5_6_15", "7_8", "9_13", "10_11_12_14"
    ]

    train_dataset_1 = TrajectoryDataset(
        dataset_dir=args.tdatapath, labels={1, 2, 3, 4, 5, 6, 15})

    train_dataset_2 = TrajectoryDataset(
        dataset_dir=args.tdatapath, labels={7, 8})

    train_dataset_3 = TrajectoryDataset(
        dataset_dir=args.tdatapath, labels={9, 13})

    train_dataset_4 = TrajectoryDataset(
        dataset_dir=args.tdatapath, labels={10, 11, 12, 14})

    train_datasets = [
        train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4
    ]

    # valid_dataset_1 = TrajectoryDataset(
    #     dataset_dir=args.vdatapath, labels={1, 2, 3, 4, 5, 6, 15})

    # valid_dataset_2 = TrajectoryDataset(
    #     dataset_dir=args.vdatapath, labels={7, 8})

    # valid_dataset_3 = TrajectoryDataset(
    #     dataset_dir=args.vdatapath, labels={9, 13})

    # valid_dataset_4 = TrajectoryDataset(
    #     dataset_dir=args.vdatapath, labels={10, 11, 12, 14})

    # valid_datasets = [
    #     valid_dataset_1, valid_dataset_2, valid_dataset_3, valid_dataset_4
    # ]

    batch_size = 1

    train_loaders = []

    valid_loaders = []

    for train_dataset in train_datasets:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size)
        train_loaders.append(train_loader)

    # for valid_dataset in valid_datasets:
    #     valid_loader = torch.utils.data.DataLoader(
    #         valid_dataset, shuffle=False, batch_size=1)
    #     valid_loaders.append(valid_loader)

    model_path = os.path.join(os.getcwd(), "models")

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    train_more_stus(train_loaders, valid_loaders, s_names,
                    args.backbone, args.epochs, args.lr)

    # calsses, losses = test(test_loader=train_loader, backbone=args.backbone)

    # plot_results(classes=calsses, losses=losses, types=15)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_false',
                        help='Turns off gpu training')
    parser.add_argument('--save_path', '-sp', type=str, default='.',
                        help='Path to a folder where metrics and models will be saved')

    parser.add_argument('--epochs', type=int, default=1000,
                        help="Train's total epoches")

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')

    parser.add_argument('--backbone', type=str,
                        default='resnet152', help='feature extract')

    parser.add_argument('--tdatapath', type=str,
                        default="/home/juxiaobing/code/GraduationProject/CNN-VAE/data/T15/train", help="dataset's path")

    parser.add_argument('--vdatapath', type=str,
                        default="/home/juxiaobing/code/GraduationProject/CNN-VAE/data/T15/test", help="dataset's path")

    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    main(args)
