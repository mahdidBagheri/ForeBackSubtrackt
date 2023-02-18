import os
from Config.DatasetConfig import width,hight
from torch.utils.data import Dataset
import pandas as pd
import cv2
class Dataloader(Dataset):
    def __init__(self, opt, csv_path):
        super(Dataloader,self).__init__()
        self.opt = opt

        path_to_scv = os.path.join(opt.root_path,opt.root_dataset, csv_path)
        self.df = pd.read_csv(path_to_scv)
        self.df = self.df.loc[self.df['scale'] > 0.6]
        a=0
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        assert index <= len(self)
        combined_path = self.df.iloc[index]["img_name"]
        mask_path = self.df.iloc[index]["mask_name"]
        position = self.df.iloc[index]["position"]
        label = self.df.iloc[index]["label"]
        print(label)
        scale = self.df.iloc[index]["scale"]

        path_to_mask = os.path.join(self.opt.root_path,self.opt.root_dataset, mask_path)
        mask_img = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img,(width,hight))

        mask_norm = cv2.normalize(mask_img,None,0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)

        path_to_combined = os.path.join(self.opt.root_path,self.opt.root_dataset, combined_path)
        combined_img = cv2.imread(path_to_combined)
        combined_img = cv2.resize(combined_img,(width,hight))
        combined_norm = cv2.normalize(combined_img,None,0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)

        print(combined_norm.shape)
        return combined_norm, mask_norm

    def test(self,index):
        input, output = self[index]
