import cv2

#import numpy as np


face_cascade = cv2.CascadeClassifier('D:/Study/Python stuff/data/haarcascades/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)


i = 0
while(i<125):
    i = i+1
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)  #for converting in grayscale
    
    faces = face_cascade.detectMultiScale(gray) #We will get x,y,w and height in face variableqq
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+h] #Region of interest
        roi_color = frame[y:y+h,x:x+h]
        j = str(i)
        path = "D:/dataset/testing/"+str(i)+".jpg"   # where pic will be save
        dim = (224,224)
        Saved_picture = cv2.resize(roi_color,dim)
        
        cv2.imwrite(path, Saved_picture ) #writing captured image in jpg file 
        
        color = (255,0,0) #blue color
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame,(x,y) , (end_cord_x,end_cord_y) , color, stroke)
    cv2.imshow('frame',frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break   
cap.release()  
cv2.destroyAllWindows()