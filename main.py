import os
import argparse
import torch
import torchvision
import torchvision.transforms as tf
from data.TrajectoriesDataSet import TrajectoryDataset
from stad.trainer.trajcestad import train


def main(args):

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = TrajectoryDataset(
        dataset_dir=args.datapath, labels={1, 2})

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size)

    model_path = os.path.join(os.getcwd(), "models")

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    train(train_loader, args.backbone, args.epochs, args.lr)


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
                        default='resnet50', help='feature extract')

    parser.add_argument('--datapath', type=str,
                        default="/home/juxiaobing/code/GraduationProject/CNN-VAE/data/T15/T15_images", help="dataset's path")

    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    main(args)
