import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.models import Model,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input
from glob import glob


IMAGE_SIZE=[224,224]


train_path="C:/Users/welcome/Desktop/All About COnvolution Networks/8782_44566_bundle_archive/flowers"
test_path="C:/Users/welcome/Desktop/All About COnvolution Networks/8782_44566_bundle_archive/test"

vgg=VGG16(input_shape=IMAGE_SIZE+ [3],weights='imagenet',include_top=False)



for layer in vgg.layers:
    layer.trainable=False
    
  
folders=glob('flowers/*')   



x=Flatten()(vgg.output)

prediction=Dense(len(folders),activation='softmax')(x)


model=Model(inputs=vgg.input,outputs=prediction)

model.summary()


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)




training_set=train_datagen.flow_from_directory('flowers',target_size=(224,224),
                                               batch_size=32,
                                               class_mode='categorical')


test_set=test_datagen.flow_from_directory('test',
                                          target_size=(224,224),
                                          batch_size=32,
                                          class_mode='categorical')


history=model.fit_generator(training_set,validation_data=test_set,epochs=5,steps_per_epoch=5,validation_steps=5)


plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='validation Accuracy')


plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='validation Loss')

import tensorflow as tf
from keras.models import load_model



model.save('new_model.h5')

