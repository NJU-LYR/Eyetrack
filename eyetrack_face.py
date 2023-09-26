import cv2
from test_time import predict,load_model
from PIL import Image
import time
import numpy as np
model=load_model()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')

# smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# 调用摄像头摄像头
cap = cv2.VideoCapture(0)
start=time.time()
count=0
vector=[]
while (True):
# for i in range(0,60):
    # 获取摄像头拍摄到的画面
    # start=time.time()
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    count+=1
    # faces = face_cascade.detectMultiScale(frame, 1.3, 2)
    img = frame
    VECTOT_PREDICT=False
    if count % 2 ==0:
        VECTOT_PREDICT=True
    eyes = eye_cascade.detectMultiScale(img, 1.3, 10)




    for (ex, ey, ew, eh) in eyes:
        
        
        if VECTOT_PREDICT==True:
            eye_area=img[ey-20:ey+eh+20,ex-20:ex+ew+20]
            # circle=cv2.HoughCircles(eye_area,cv2.HOUGH_GRADIENT,1,20,)
            predict_area=Image.fromarray(eye_area)
            vector=predict(predict_area,model)
            eye_area=cv2.resize(eye_area,(224,224))

            # cv2.line(eye_area, (int(ew/2),int(eh/2)), (int(ew/2+np.array(-vector[0][0]*60).astype(int)),int(eh/2+np.array(-vector[0][1]*60).astype(int))) , (0, 255, 255), 2)
            # eye_area=cv2.resize(eye_area,(224,224))
            cv2.circle(img,(int(ex+ew*0.5),int(ey+eh*0.5)),2,(0,255,0))
            # cv2.line(img, (int(ex+ew*0.5),int(ey+eh*0.5)), (int(ex+ew/2+np.array(-vector[0][0]*80).astype(int)),int(ey+eh/2+np.array(-vector[0][1]*80).astype(int))) , (0, 255, 255), 2)
            cv2.imshow('eye',eye_area)
        # img=cv2.putText(img,",".join(str(i) for i in vector),(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
            
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