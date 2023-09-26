import json
from glob import glob
import time
import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from model import resnet18
import numpy as np
from customdataset import CustomDataset
import cv2
from torch.ao.quantization.fx.graph_module import ObservedGraphModule



def write2json():
    json_fns_test = glob("./test_imgs/*.json")
    num = 0
    cla_dict = {}
    for json_fn in json_fns_test:
        num += 1
        name = "".join(list(filter(str.isdigit, json_fn)))
        with open(json_fn) as data_file:
            data = json.load(data_file)
            look_vec = list(eval(data['eye_details']['look_vec']))
            pic_name = name + '.jpg'
            cla_dict[pic_name] = look_vec[:3]
    # print(cla_dict.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices_test.json', 'w') as json_file:
        json_file.write(json_str)
    print("数据总数为：{},{}".format(num, num))


def load_model():
    device = torch.device("cpu")
    # device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #load image
    #指向需要遍历预测的图像文件夹
    imgs_root = "./test_imgs"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    #读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    #read class_indict
    json_path = './class_indices_test.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    # print(class_indict["%s.jpg" % 1])
    cla_list=list(class_indict.values())
    # print(cla_list)
    #create model
    model = resnet18()
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 3)    # 输入层数：in_channel 输出层数：3（x，y，z）!
    model.to(device)

    #load model weights
    weights_path = "./resNet18.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    #prediction
    model.eval()
    return model

def predict(img,model):
    device=torch.device("cpu")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(360),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

   
    with torch.no_grad():

        img = data_transform(img)
        img = img.unsqueeze(0)
        output = model(img)#.to(device)).cpu()
        output_list=output.tolist()
        output_list=np.round(output_list,3)
            
    return output_list
if __name__ == '__main__':
    # cv2.imshow('eye',testimg)
    # cv2.waitKey(0)
    # print(testimg.type)
    
    # write2json()
    img1_pre=Image.open("3.jpg")
    # img1.show()
    model=load_model()
    t1=time.time()
    result1=predict(img1_pre,model)
    print(result1)
    t2=time.time()
    img1=cv2.imread("3.jpg")
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    print('运行时间：%.3fs' % (t2-t1))
    img2=cv2.imread("1.jpg")
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    img2_pre=Image.open("1.jpg")
    t3=time.time()
    result=predict(img2_pre,model)
    print(result)
    t4=time.time()
    cv2.line(img1,(360,320),(360+int(100*result1[0,0]),320+int(-100*result1[0,1])),(255,0,0),2)
    # cv2.imshow("1",img1)
    img1=Image.fromarray(img1)
    img1.show()
    img2=Image.fromarray(img2)
    # img2.show()
    print('运行时间：%.3fs' % (t3-t4))
