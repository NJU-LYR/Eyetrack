import cv2
import matplotlib.pyplot as plt


img=cv2.imread('3.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst_img = cv2.medianBlur(img_gray, 7)

circle = cv2.HoughCircles(dst_img, cv2.HOUGH_GRADIENT, 1, 30,
                         param1=50, param2=30, minRadius=0, maxRadius=50)

for i in circle[0, :]:  # 遍历矩阵的每一行的数据
    # 绘制圆形
    cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 2)

plt.imshow(img[:,:,::-1])
plt.show()