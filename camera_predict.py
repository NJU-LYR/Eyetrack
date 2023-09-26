import cv2 as cv
import torch
from torchvision import transforms
from model import resnet18
"""
OpenCV打开摄像头方式：
    https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
"""


def axis_predict(videos):
    device = "cpu"   # GPU: cuda:0
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.CenterCrop(224),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    videos = data_transform(videos)
    videos = torch.unsqueeze(videos, dim=0)  # 普通的图像文件有3维度，需要增加一个Batch维度：设置第一个维度为0
    # ----------------------------------------------------------------------------------
    model = resnet18(num_classes=3).to(device)  # 可修改的网络架构
    weights_path = "./resNet18.pth"  # 权重文件
    # ----------------------------------------------------------------------------------
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(videos.to(device))).cpu()
    print("predict [x, y, z]: {}".format(output))


def catch_camera(name='my_camera', camera_index=2):
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        raise Exception('Cannot open the camera.')

    while cap.isOpened():
        loop_start = cv.getTickCount()
        ret, frame = cap.read()  # if frame is read correctly ret is true
        if not ret:
            print("Can't receive frame. Exiting ··· ")
            break

        # cv.circle(frame, (480, 270), 3, (255, 0, 0), 3)
        cv.imshow(name, frame) # 在window上显示图片
        # 利用模型进行预测
        axis_predict(frame)
        key = cv.waitKey(10)
        loop_time = cv.getTickCount() - loop_start
        total_time = loop_time/(cv.getTickFrequency())
        running_FPS = int(1/total_time)
        print("Running FPS: {}".format(running_FPS))

        if key & 0xFF == ord('q'):
            # 按q退出
            break
        if cv.getWindowProperty(name, cv.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
    # 释放摄像头
    cap.release()
    cv.destroyAllWindows()


def main():
    catch_camera(camera_index=1)


if __name__ == '__main__':
    main()
