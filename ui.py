from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QImage, QPixmap
import cv2
import sys
from test_time import load_model,predict
from PIL import Image
 
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(753, 629)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(210, 520, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 20, 640, 480))
        self.label.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(440, 520, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 753, 26))
        self.menubar.setObjectName("menubar")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(210, 560, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(440, 560, 93, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
 
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
 
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "打开摄像头"))
        self.pushButton_2.setText(_translate("MainWindow", "关闭摄像头"))
        self.pushButton_3.setText(_translate("MainWindow", "加载模型"))
        self.pushButton_4.setText(_translate("MainWindow", "视线预测"))

class Open_Camera(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(Open_Camera,self).__init__()
        self.setupUi(self) 
        self.init()
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) #摄像头
        self.model=load_model()
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
    def init(self):
        #定时器让其定时读取显示图片
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.show_image)
        #打开摄像头
        self.pushButton.clicked.connect(self.open_camera)
        #关闭摄像头
        self.pushButton_2.clicked.connect(self.close_camera)
    #开启摄像头
    def open_camera(self):
        if self.cap.isOpened():
            self.camera_timer.start(40) 
            self.show_image()
        else:
            QMessageBox.critical(self,'错误','摄像头未打开！')
            return None
    #显示图片
    def show_image(self):
        flag,image = self.cap.read()  
        image_show = cv2.resize(image,(640,400))  #把读到的帧的大小重新设置为 1280x800
        width,height = image_show.shape[:2] #行:宽，列:高
        image_show = cv2.cvtColor(image_show,cv2.COLOR_BGR2RGB)  
        image_show = cv2.flip(image_show, 1)  #水平翻转，因为摄像头拍的是镜像的。
        eyes = self.eye_cascade.detectMultiScale(image_show, 1.3, 10)
        vector=[]
        for (ex, ey, ew, eh) in eyes:
            eye_area=image_show[ey-20:ey+eh+20,ex-20:ex+ew+20]
            # circle=cv2.HoughCircles(eye_area,cv2.HOUGH_GRADIENT,1,20,)
            predict_area=Image.fromarray(eye_area)
            vector=predict(predict_area,self.model)

            cv2.circle(image_show,(int(ex+ew*0.5),int(ey+eh*0.5)),2,(0,255,0))
            # cv2.line(img, (int(ex+ew*0.5),int(ey+eh*0.5)), (int(ex+ew/2+np.array(-vector[0][0]*80).astype(int)),int(ey+eh/2+np.array(-vector[0][1]*80).astype(int))) , (0, 255, 255), 2)
            # img=cv2.putText(img,",".join(str(i) for i in vector),(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
            cv2.rectangle(image_show, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
            
        image_show=cv2.putText(image_show,",".join(str(i) for i in vector),(320,200),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)    
        show = QtGui.QImage(image_show.data,height,width,QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(show))  
    #关闭摄像头
    def close_camera(self):
        self.camera_timer.stop() #停止读取
        self.cap.release() #释放摄像头
        self.label.clear() #清除label组件上的图片
 
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Open_Camera()
    ui.show()
    sys.exit(app.exec_())