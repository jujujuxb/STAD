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

    train_dataset = TrajectoryDataset(
        dataset_dir=args.vdatapath, labels={1, 2, 3})

    batch_size = 1

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size)

    model_path = os.path.join(os.getcwd(), "models")

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    calsses, losses = test(test_loader=train_loader,
                           backbone=args.backbone, threahold=6.0)

    plot_results(classes=calsses, losses=losses, lists={
                 1, 2, 3}, types=15)

    tt = sorted(losses)

    c = (int)(len(losses) * 0.05)

    print(tt[-1], tt[-c])

    pass


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
                        default="/home/juxiaobing/code/GraduationProject/CNN-VAE/data/T15/anatation_images", help="dataset's path")

    # /home/juxiaobing/code/GraduationProject/CNN-VAE/data/T15/anatation_images

    # /home/juxiaobing/code/GraduationProject/CNN-VAE/data/T15/test

    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    main(args)
