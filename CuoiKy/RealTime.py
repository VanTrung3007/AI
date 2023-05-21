import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

model1 = load_model('train.h5')

# open webcam
webcam = cv2.VideoCapture(0)

# detectface
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# class 
classes = ['BuonNgu','TinhTao']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    grayface = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #Chuyển ảnh sang mức xám
    faces = face_cascade.detectMultiScale(grayface)         #Tách khuôn mặt

    for (x,y,w,h) in faces:
        #Tạo 4 giá trị để tạo hình vuông tập trung vào khuôn mặt
        startX = x
        startY = y
        endX = x + w
        endY = y + h

        #Vẽ hình vuông lên ảnh
        cv2.rectangle(frame, (startX-10,startY-10), (endX+10,endY+10), (0,255,0), 2)

        #Tạo 1 bản sao cho khuôn mặt vừa được tách
        face_crop = np.copy(frame[startY:endY,startX:endX])

        #Resize bản sao
        face_crop = np.array(face_crop)
        face_crop_iden = tf.image.resize(face_crop, [150,150])
        face_crop_iden = np.expand_dims(face_crop_iden, axis=0)
        
        #Máy dự đoán
        iden = model1.predict(face_crop_iden)[0]

        #Tạo biến để nhận kết quả dự đoán của máy
        idx = np.argmax(iden)
        label1 = classes[idx]
        #Tạo chữ hiển thị trên khung ảnh
        label = "{}".format(label1)
        #Vị trí xuất hiện của chữ
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        #Cho hiển thị chữ lên khung ảnh
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    #Hiển thị hình ảnh đã xử lí
    cv2.imshow("Drowsy", frame)

    #Nhấn phím "Q" để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Giải phóng dữ liệu
webcam.release()
cv2.destroyAllWindows()