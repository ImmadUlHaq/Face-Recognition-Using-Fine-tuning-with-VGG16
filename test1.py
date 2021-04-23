# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 01:09:51 2021

@author: Immad
"""

import os
import cv2
import numpy as np
from os import listdir
from keras.models import load_model
from PIL import Image
import datetime 
import mysql.connector
mydb = mysql.connector.connect(host = "localhost", user = "root", database = "fyp")
mycursor = mydb.cursor()

from keras.preprocessing import image

from keras.models import load_model

classifier = load_model('D:/Ethics/face_recognize6_vgg16.h5')


from os.path import isfile, join
face_recognize_dict = {"[0]": "Adnan", 
                      "[1]": "Immad",
                      "[2]": "Talha"}

face_recognize_dict_enroll = {"[0]": 3, 
                      "[1]": 51,
                      "[2]": 49}

def draw_test(name, pred, im):
    
    face_cascade = cv2.CascadeClassifier('D:/Study/Python stuff/data/haarcascades/haarcascade_frontalface_alt2.xml')
    faces_found = face_cascade.detectMultiScale(im, 1.3, 5)
    
    if faces_found is ():
        face = "No face found"
    
    for (x,y,w,h) in faces_found:
        cv2.rectangle(im, (x,y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = im[y:y+h, x:x+w]
        
        if (classlabel[0][0] > 0.5 or classlabel[0][1] > 0.5 or classlabel[0][2]<0.5):
            face = face_recognize_dict[str(np.argmax(pred, axis=1))]
            ID = face_recognize_dict_enroll[str(np.argmax(pred, axis=1))]
            today = datetime.datetime.now()
        
            #Here we are applying database coding#
            try:
                s = "insert into attendence(enroll_id, name, time, present) values (%s,%s,%s,%s)"
                b1 = (ID, face,today,True)
                mycursor.execute(s,b1)
            except:
                print('Already present')
        
        else:
            face = "No match found"
            ID = 3
            today = datetime.datetime.now()
            
            try:
                s = "insert into visitors(ID, name, date, present) values (%s,%s,%s,%s)"
                b1 = (ID, face,today,True)
                mycursor.execute(s,b1)
            except:
                print('Already present')
            
    
    Gray = [100,100,100]
    expanded_image = cv2.copyMakeBorder(im, 60, 0, 0, 0 ,cv2.BORDER_CONSTANT,value=Gray)
    cv2.putText(expanded_image, face, (20, 40) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255), 1)
    cv2.imshow(name, expanded_image)
    return
    

   
    
video_capture = cv2.VideoCapture(0)
while True:
    
    _,input_im = video_capture.read()

    
    input_original = input_im.copy()
    
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    
    input_im = input_im.reshape(1,224,224,3) 
    
         # Get Prediction
    classlabel = classifier.predict(input_im, 1, verbose = 0)     
    
    print(classlabel)
   
         # Show image with predicted class
    draw_test("Prediction", classlabel, input_original)
   
    if cv2.waitKey(1)==13:
        break
cv2.destroyAllWindows()
video_capture.release()  
#cap.release()