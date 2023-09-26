import cv2
 
cap = cv2.VideoCapture(0)
 
if not (cap.isOpened()):
    print("Could not open video device")
 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
 
while(True):    
    # Capture frame-by-frame    
    ret, frame = cap.read()    
    
    cv2.imshow('frame', frame)
    
    #Waits for a user input to quit the application    
    if cv2.waitKey(1) & 0xFF == ord('q'):    
        break
    
cap.release()
cv2.destroyAllWindows()