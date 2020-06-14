from keras.models import load_model


from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
model=load_model('new_model.h5')
img=image.load_img('C:/Users/welcome/Desktop/All About COnvolution Networks/8782_44566_bundle_archive/112650879_82adc2cc04_n.jpg',target_size=(224,224))

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data) 