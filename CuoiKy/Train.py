import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

train_data=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_set=train_data.flow_from_directory('Train',target_size=(150,150), batch_size=32, class_mode='categorical')
test_set=test_data.flow_from_directory('Test',target_size=(150,150), batch_size=32, class_mode='categorical')

print(training_set.class_indices)
print(test_set.class_indices)

# Tạo ra mạng CNN để train mô hình
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu',  padding='same',input_shape=(150,150,3))) 
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,kernel_size=(3,3), activation='relu',  padding='same')) 
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,kernel_size=(3,3), activation='relu',  padding='same')) 
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(2,activation='softmax'))
model.summary()

# Biên dịch 
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Gán biến lại để vẽ đồ thị, với 30 lần học (epochs), mỗi lần học thì chỉ học␣128 dữ liệu (batch_size) 
train=model.fit(training_set,validation_data=test_set,epochs=30,batch_size=128,verbose=1)

# Đánh giá độ chính xác của mô hình 
Score=model.evaluate(training_set,verbose=0)
print('Train Loss', Score[0])
print('Train Accuracy', Score[1])

model.save('train.h5')