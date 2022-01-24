import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CatSegmentationDataset as Dataset
from loss import DiceLoss
from unet import UNet


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train = data_loaders(args)

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)

    dsc_loss = DiceLoss()

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    #logger = Logger(args.logs)
    loss_train = []

    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        unet.train()

        for i, data in enumerate(loader_train):
            step += 1

            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = unet(x)
            optimizer.zero_grad()
            loss = dsc_loss(y_pred, y_true)

            loss_train.append(loss.item())
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                print('Step ', step, 'Loss', np.mean(loss_train))
                loss_train = []

        torch.save(unet, args.ckpts + '/unet_epoch_{}.pth'.format(epoch))


def data_loaders(args):
    dataset_train = Dataset(
        images_dir=args.images,
        image_size=args.image_size,
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    return loader_train


def makedirs(args):
    os.makedirs(args.ckpts, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of Cat"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch Size (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epoch number (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Workers' count (default: 4)",
    )
    parser.add_argument(
        "--ckpts", type=str, default="./ckpts", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="./data", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    args = parser.parse_args()
    main(args)
