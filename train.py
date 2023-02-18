import os

import torch
import torchvision
from dataset import Dataloader
import argparse

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--root_path", default=os.getcwd())
    parser.add_argument("--root_dataset", required=True)
    opt = parser.parse_args()

    train_dataset = Dataloader.Dataloader(opt, opt.train_csv)
    test_dataset = Dataloader.Dataloader(opt, opt.test_csv)

    while True:
        v = int(input())
        train_dataset.test(v)