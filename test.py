import numpy as np
import torch
import torchvision
import argparse
import cv2
from Config.DatasetConfig import width,hight
from torchvision.transforms import transforms
from models import Unet

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--img_path")
    parser.add_argument("--save", default="cat.png")

    opt = parser.parse_args()

    model = Unet.UNET()
    model.load_state_dict(torch.load(opt.model))
    model.eval()

    img = cv2.imread(opt.img_path)

    combined_img = cv2.resize(img, (width, hight))
    combined_norm = cv2.normalize(combined_img, None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    # combined_norm_t = torch.from_numpy(combined_norm)
    transform_list_output = [transforms.ToTensor()]
    output_transform = transforms.Compose(transform_list_output)

    combined_norm_t = output_transform(combined_norm)[None,:,:,:]

    if(torch.cuda.is_available()):
        combined_norm_t = combined_norm_t.cuda()
        model = model.cuda()

    with torch.no_grad():
        output = model(combined_norm_t)


    output = output.cpu().detach().numpy().reshape(combined_img.shape[0],combined_img.shape[1],1)
    result = cv2.normalize(output,None,0,255,cv2.NORM_MINMAX, cv2.CV_8U)
    _, result = cv2.threshold(result,40,255,cv2.THRESH_BINARY)

    result = cv2.resize(result,(img.shape[1],img.shape[0]))
    result = np.expand_dims(result, axis=2)

    result_as_B = np.concatenate((result, np.zeros((result.shape[0],result.shape[1],1)),np.zeros((result.shape[0],result.shape[1],1))),axis=2)
    result_as_B = result_as_B.astype(np.uint8)
    overlay = cv2.addWeighted(img,1.0,result_as_B,0.5, gamma=1)
    cat = np.concatenate((img,result_as_B,overlay), axis=1)
    cv2.imwrite(opt.save, cat)
    # cv2.imshow("source", combined_img)
    # cv2.imshow("mask",result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()