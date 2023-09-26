import os
import json
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
# import torchvision.transforms as transforms


class CustomDataset(Dataset):

    def __init__(self, annotations_file, img_dir, transform=None,   # annotations_file : 标注文件 # image_dir 图像文件地址
                 target_transform=None):
        self.img_labels = pd.read_json(annotations_file).T
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        with open(annotations_file) as F:
            self.dic_json = json.load(F)  # 利用库json加载.json文件
            self.img_name_list = list(self.dic_json.keys())  # 利用json来获取列表, 打开文件获取列表放在__init__是因为只需要初始化的时候执行一次即可！

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_name_list[idx]) # 读P取文件是没有问题的---路径也是正确的！
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # 获取图像路径
        # iloc[a, b]函数是取行索引为a,列索引为b的数据！
        # image = read_image(img_path)  # --------------这一步出现了问题！  ----------------
        #totensor = transforms.ToTensor()
        image = Image.open(img_path)

        label = list(self.img_labels.iloc[idx, :])  # 将xyz坐标转换成list类型[x,y,z]
        if self.transform:  # 将图片(ndarray)转换成tensor格式！
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # print("idx: {}".format(idx))
        return image, label
    """
       .json文件类似于：
       1.jpg [x, y, z, 0]
       2.jpg [x, y, z, 0]
       ······
    """

