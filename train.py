import os

import torch
import torchvision
from dataset import Dataloader
import argparse

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_csv", required=True)
    parser.add_argument("-t", "--test_csv", required=True)
    parser.add_argument("-r", "--root_path", default=os.getcwd())
    opt = parser.parse_args()

    train_dataset = Dataloader.Dataloader(opt)