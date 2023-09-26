import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from customdataset import CustomDataset
import torch.nn as nn
from model import resnet18
import time
import copy
from torch.quantization import get_default_qconfig
import matplotlib.pyplot as plt
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.fx.graph_module import ObservedGraphModule








def main():
    # t1=time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")
    # device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def calib_quant_model(model, calib_dataloader):
        """
        校准函数
        """
        assert isinstance(
            model, ObservedGraphModule
        ), "model must be a perpared fx ObservedGraphModule."
        model.eval()
        with torch.inference_mode():
            for inputs, labels in calib_dataloader:
                model(inputs.to(device))
        print("calib done.")

    def quant_fx(model,calibdata):
        """
        使用Pytorch中的FX模式对模型进行量化
        """
        model.eval()
        qconfig = get_default_qconfig("fbgemm")  # 默认是静态量化
        qconfig_dict = {
            "": qconfig
            # 'object_type': []
        }
        model_to_quantize = copy.deepcopy(model)
        model_to_quantize.eval()
        example_inputs = (torch.randn(1, 3, 224, 224))
        prepared_model = prepare_fx(model_to_quantize, qconfig_dict,example_inputs)
        print("prepared model: ", prepared_model.graph)
        calib_quant_model(prepared_model,calibdata)
        quantized_model = convert_fx(prepared_model)
        print("quantized model: ", quantized_model)
        torch.save(model.state_dict(), "r18.pth")
        torch.save(quantized_model.state_dict(), "r18_quant.pth")

    data_transform = transforms.Compose(
        [transforms.Resize(360),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = "./test_imgs"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = './class_indices_test.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    # print(class_indict["%s.jpg" % 1])
    cla_list=list(class_indict.values())
    # print(cla_list)
    # create model
    model = resnet18()
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 3)    # 输入层数：in_channel 输出层数：3（x，y，z）!
    # model.to(device)



    data_root = os.path.abspath(os.path.join(os.getcwd(), "test_imgs"))
    image_path = os.path.join(data_root, "imgs")
    batch_size = 32
    nw = 0
    annotations_val_file = os.path.abspath(os.path.join(os.getcwd(), "class_indices_test.json"))
    validate_dataset = CustomDataset(annotations_val_file,data_root,
                                     transform=data_transform)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)

    # load model weights
    weights_path = "./resNet18.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    state_dict=torch.load(weights_path, map_location=device)
    # qconfig = get_default_qconfig("fbgemm")  # 默认是静态量化
    # qconfig_dict = {
    #     "": qconfig
    #     # 'object_type': []
    # }
    # example_inputs = (torch.randn(1, 3, 224, 224))
    # model_32 = copy.deepcopy(model)
    # model_32.eval()
    # model_32_pre = prepare_fx(model_32,qconfig_dict,example_inputs)
    # model_32_pre.to(device)
    # calib_quant_model(model_32_pre,validate_loader)
    
    # model_8 = convert_fx(model_32_pre)
    # model_8.load_state_dict(state_dict)
    # model=model_8
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # model.eval()

    # quant_fx(model)
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "test_imgs"))
    # image_path = os.path.join(data_root, "imgs")
    # batch_size = 32
    # nw = 0
    # annotations_val_file = os.path.abspath(os.path.join(os.getcwd(), "class_indices_test.json"))
    # validate_dataset = CustomDataset(annotations_val_file,data_root,
    #                                  transform=data_transform)
    # validate_loader = torch.utils.data.DataLoader(validate_dataset,
    #                                               batch_size=batch_size, shuffle=True,
    #                                               num_workers=nw)
    # quant_fx(model,validate_loader)
    # calib_quant_model(model,validate_loader)


    # prediction
    # model.eval()
    t1=time.time()
    batch_size = 16  # 每次预测时将多少张图片打包成一个batch
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            output_list=output.tolist()
            # predict = torch.softmax(output, dim=1)
            # probs, classes = torch.max(predict, dim=1)
            # print(output)
            # 计算误差
            loss_list=[]
            for idx in range(0,batch_size):
                
                loss=0
                for i in range(0,3):
                    loss+=(cla_list[ids * batch_size + idx][i]-output_list[idx][i])*(cla_list[ids * batch_size + idx][i]-output_list[idx][i])
                    loss=loss/3
                    loss_list.append(loss)
                # print(img_path_list)
                # print('%s.jpg' % str(ids * batch_size + idx+1))
                # print(type(class_indict['%s.jpg' % str(ids * batch_size + idx+1)]))
                # print(output[ids * batch_size + idx].tolist())
                print("image:{0}  real_vector:{1[0]},{1[1]},{1[2]}  predict_vector:{2[0]:.4},{2[1]:.4},{2[2]:.4}  loss:{3:.3}".format(
                                                                img_path_list[ids * batch_size + idx],
                                                                cla_list[ids * batch_size + idx],
                                                                output_list[idx],
                                                                loss_list[idx]))
    t2=time.time()
    print('运行时间：%.3fs' % (t2-t1))

if __name__ == '__main__':
    main()
