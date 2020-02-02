import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,MaxPooling2D,Dropout,Conv2D,Flatten,Activation
from tensorflow.keras.models import Sequential
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


#train data preprocessing
path = "v_data/train/"
train_images = []
train_labels = []
image_size = (128,128)
for i in os.listdir(path):
  labl = i    #copying class name to labl variable
  i = path+i
  for j in os.listdir(i):
    j = i+"/"+j
    img = Image.open(j).convert('LA')
    img = img.resize(image_size)
    data = np.asarray(img,dtype="float32")
    train_images.append(data)
    if(labl=="cars"):
      train_labels.append(0)
    else:
      train_labels.append(1)
train_images = np.array(train_images)
train_labels = np.array(train_labels)
# print(len(train_images))
# print(len(train_labels))

#test data preprocessing
path = "v_data/test/"
test_images = []
test_labels = []
image_size = (128,128)
for i in os.listdir(path):
  labl = i    #copying class name to labl variable
  i = path+i
  for j in os.listdir(i):
    j = i+"/"+j
    img = Image.open(j).convert('LA')
    img = img.resize(image_size)
    data = np.asarray(img,dtype="float32")
    test_images.append(data)
    if(labl=="cars"):
      test_labels.append(0)
    else:
      test_labels.append(1)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
# print(len(test_images))
# print(len(test_labels))


# Model Architecture
model = Sequential() 
model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
# model.add(Conv2D(64, (2, 2))) 
# model.add(Activation('relu')) 
# model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid'))

#Compile model
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

#Fit the model
model.fit(x=train_images,y=train_labels,epochs=10,validation_data=(test_images,test_labels),shuffle=True,batch_size=16)