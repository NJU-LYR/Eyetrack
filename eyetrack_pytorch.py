import cv2
from test_time import predict,load_model
from PIL import Image
import time
import numpy as np
model=load_model()


# smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# 调用摄像头摄像头
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
cap_show = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap_show.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap_show.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap_show.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
start=time.time()
count=0
vector=[]
i=0
while (True):
# for i in range(0,60):
    # 获取摄像头拍摄到的画面
    # start=time.time()
    ret, frame = cap.read()
    frame=cv2.flip(frame,-1)
    frame=cv2.resize(frame,(800,600))
    count+=1
    # faces = face_cascade.detectMultiScale(frame, 1.3, 2)
    img = frame
    VECTOT_PREDICT=False
    if count % 1 ==0:
        VECTOT_PREDICT=True

    if VECTOT_PREDICT==True:
        predict_area=Image.fromarray(frame)
        vector=predict(predict_area,model)
        eye_area=cv2.resize(frame,(224,224))

        # cv2.line(eye_area, (int(ew/2),int(eh/2)), (int(ew/2+np.array(-vector[0][0]*60).astype(int)),int(eh/2+np.array(-vector[0][1]*60).astype(int))) , (0, 255, 255), 2)
        # eye_area=cv2.resize(eye_area,(224,224))
        # cv2.line(img, (int(ex+ew*0.5),int(ey+eh*0.5)), (int(ex+ew/2+np.array(-vector[0][0]*80).astype(int)),int(ey+eh/2+np.array(-vector[0][1]*80).astype(int))) , (0, 255, 255), 2)
        # cv2.imshow('eye',eye_area)

    #单目测距
    _, frame_show= cap_show.read()
    frame_show=cv2.flip(frame_show,-1)
    # frame = cv.imread("d:/files/DMS/DMS/test/1/1.jpg",1)
    hsv = cv2.cvtColor(frame_show, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
    sensitivity = 15
    red0 = [0, 50, 50]
    red1 = [20, 255, 255]
    lower = np.array(red0)
    upper = np.array(red1)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 3)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame_show, frame_show, mask=mask)
    D = 0
    # 查找最大的红色色块
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (x0, y0, x1, y1) = (1000, 1000, 0, 0)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        print("x:", x, "y:", y, "w:", w, "h:", h)
        if w * h < 100:
            continue
        if x < x0:
            x0 = x
        if y < y0:
            y0 = y
        if x + w > x1:
            x1 = x + w
        if y + h > y1:
            y1 = y + h
        # print(x, y, w, h, w * h)
    if (x1 - x0) * (y1 - y0) > 600:
        cv2.rectangle(res, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.rectangle(frame_show, (x0, y0), (x1, y1), (0, 0, 255), 2)
        D = int(3.5 * 198 / ((x1 - x0) * 5.3 / 800))  # 测距公式
        print("w:", (x1 - x0), "h:", (y1 - y0), "i:", i, "距离：", D)
    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame_show)

    if 1:
        _, frame= cap.read()
        frame = cv2.flip(frame,-1)
        # pre_x = np.argmax(pre[0]) / 10 - 0.5
        # pre_y = np.argmax(pre[1]) / 10 - 0.5
        # pre_y = -pre_y
        #out.write(frame)
        # frame = cv.add(img, frame)  # 两个参数1代表加权系数（融合比例），1代表最大
        frame=cv2.resize(frame,(800,600))
        frame[0:30, 0:30, 2] = 255
        frame[0:30, 0:30, 0:2] = 0
        frame[0:30, 770:800, 2] = 255
        frame[0:30, 770:800, 0:2] = 0
        frame[285:315, 0:30, 0:2] = 0
        frame[285:315, 0:30, 2] = 255
        frame[285:315, 770:800, 0:2] = 0
        frame[285:315, 770:800, 2] = 255
        frame[570:600, 0:30, 2] = 255
        frame[570:600, 0:30, 0:2] = 0
        frame[570:600, 770:800, 0:2] = 0
        frame[570:600, 770:800, 2] = 255
        frame[290:310, 390:410, 0:2] = 255
        # z = math.sqrt(1 - pre_x * pre_x - pre_y * pre_y)
        # length = (D + 50) / z
        # x = round(length * pre_x * 960 / 198)
        # y = round(length * pre_y * 540 / 111)
        cv2.circle(frame, (400, 300), 6, (0, 255, 255), -1)
        cv2.line(frame, (400, 300), (400 + int(vector[0,0] * 80), 300 + int(-vector[0,1] * 80)), (0, 0, 0), 3)

        #det.write(str(i) + "_" + str(480 + x) + "_" + str(270 + y) + "_" + str(D + 50) + '\n')
        cv2.imshow("window1", frame)
    
    img=cv2.putText(img,",".join(str(i) for i in vector),(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)    
    end=time.time()
    fps=count/(end-start)
    img=cv2.putText(img,'fps:%.3f'% fps,(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)

    # 实时展示效果画面
    # cv2.imshow('eye',eye_area)
    cv2.imshow('frame2', img)
    # 每5毫秒监听一次键盘动作
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('frame2', cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
        break
# end=time.time()
# fps=60/(end-start)
# print(fps)
# 最后，关闭所有窗口
cap.release()
cv2.destroyAllWindows()