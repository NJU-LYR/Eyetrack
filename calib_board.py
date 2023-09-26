import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def calib_board():


    img = np.ones((600, 800, 3), np.uint8)
    img *=255
    cv2.line(img,(0,200),(800,200),(0,0,0),1)
    cv2.line(img,(0,400),(800,400),(0,0,0),1)

    cv2.line(img,(265,0),(265,600),(0,0,0),1)
    cv2.line(img,(530,0),(530,600),(0,0,0),1)

    cv2.circle(img,(132,100),4,(0,0,255),-1)
    cv2.circle(img,(132,300),4,(0,0,255),-1)
    cv2.circle(img,(132,500),4,(0,0,255),-1)
    cv2.circle(img,(397,100),4,(0,0,255),-1)
    cv2.circle(img,(397,300),4,(0,0,255),-1)
    cv2.circle(img,(397,500),4,(0,0,255),-1)
    cv2.circle(img,(663,100),4,(0,0,255),-1)
    cv2.circle(img,(663,300),4,(0,0,255),-1)
    cv2.circle(img,(663,500),4,(0,0,255),-1)

    return img

def distance_board():
    frame=np.ones((900, 1600, 3), np.uint8)
    frame *=255
    frame[0:100, 0:100, 1] = 0
    frame[0:100, 0:100, 0] = 0
    frame[0:100, 0:100, 2] = 255
    frame[0:100, 1500:1600, 1] = 0
    frame[0:100, 1500:1600, 0] = 0
    frame[0:100, 1500:1600, 2] =255
    frame[800:900, 0:100, 1] = 0
    frame[800:900, 0:100, 0] = 0
    frame[800:900, 0:100, 2] = 255
    return frame

def distance_board_circle():
    frame=np.ones((900, 1600, 3), np.uint8)
    frame *=255
    cv2.circle(frame,(50,50),50,(0,255,0),-1)
    cv2.circle(frame,(1550,50),50,(0,255,0),-1)
    cv2.circle(frame,(50,850),50,(0,255,0),-1)
    return frame



def output_calib_data(x,y,d,path):
    with open(path,'a') as writers:
        writers.write(str(x)+','+str(y)+','+str(d)+'\n')
        











if __name__ == '__main__':

    img=calib_board()
    path='./calib_data/'
    dis=distance_board()

    while(True):
        cv2.imshow("dis",dis)
      
       
        cv2.imshow("img",img)
        if cv2.waitKey(2) & 0xFF == ord('s'):
            print("开始标定")
            cv2.circle(img,(132,100),8,(255,0,0),2)
            cv2.imshow("img",img)
            cv2.waitKey(5000)
            cv2.circle(img,(132,100),8,(255,255,255),2)
            cv2.circle(img,(132,300),8,(255,0,0),2)
            cv2.imshow("img",img)
            cv2.waitKey(5000)
            cv2.circle(img,(132,300),8,(255,255,255),2)
            cv2.circle(img,(132,500),8,(255,0,0),2)
            cv2.imshow("img",img)
            cv2.waitKey(5000)
            cv2.circle(img,(132,100),8,(255,255,255),2)
            cv2.circle(img,(132,300),8,(255,0,0),2)
            cv2.imshow("img",img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)