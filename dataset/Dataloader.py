from torch.utils.data import Dataset
import pandas as pd
import cv2
class Dataloader(Dataset):
    def __init__(self, opt):
        super(Dataloader,self).__init__()
        self.df = pd.read_excel(opt.csv_path)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        assert index <= len(self)
        combined_path = self.df[index]["img_name"]
        foreground_path = self.df[index]["mask_name"]
        position = self.df[index]["position"]
        label = self.df[index]["position"]
        scale = self.df[index]["scale"]

        foreground_img = cv2.imread(foreground_path)

        foreground_resized = cv2.resize(foreground_img,(int(foreground_img.shape[0]*scale),int(foreground_img.shape[0]*scale)))

    def test(self,index):
        assert index <= len(self)
        combined_path = self.df[index]["img_name"]
        foreground_path = self.df[index]["mask_name"]
        position = self.df[index]["position"]
        label = self.df[index]["position"]
        scale = self.df[index]["scale"]

        foreground_img = cv2.imread(foreground_path)

        foreground_resized = cv2.resize(foreground_img,
                                        (int(foreground_img.shape[0] * scale), int(foreground_img.shape[0] * scale)))
