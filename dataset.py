import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image 


class CatSegmentationDataset(Dataset):
    
    # 模型输入是3通道数据
    in_channels = 3
    # 模型输出是1通道数据
    out_channels = 1

    def __init__(
        self,
        images_dir,
        image_size=32,
    ):

        print("Reading images...")
        # 原图所在的位置
        image_root_path = images_dir + os.sep + 'JPEGImages'
        # Mask所在的位置
        mask_root_path = images_dir + os.sep + 'SegmentationClassPNG'
        # 将图片与Mask读入后，分别存在image_slices与mask_slices中
        self.image_slices = []
        self.mask_slices = []
        for im_name in os.listdir(image_root_path):
            
            mask_name = im_name.split('.')[0] + '.png' 

            image_path = image_root_path + os.sep + im_name
            mask_path = mask_root_path + os.sep + mask_name

            im = np.asarray(Image.open(image_path).resize((image_size, image_size)))
            mask = np.asarray(Image.open(mask_path).resize((image_size, image_size)))
            self.image_slices.append(im / 255.)
            self.mask_slices.append(mask)

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):

        image = self.image_slices[idx] 
        mask = self.mask_slices[idx] 

        image = image.transpose(2, 0, 1)
        mask = mask[np.newaxis, :, :]

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        return image, mask
