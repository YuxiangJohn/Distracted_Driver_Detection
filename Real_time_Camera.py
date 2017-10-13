
# coding: utf-8

# In[5]:

from keras.preprocessing import image as image_utils
#from imagenet_utils import decode_predictions
#from imagenet_utils import preprocess_input
#from vgg16 import VGG16
import argparse
import cv2
import numpy as np
import os
import random
import sys

import threading

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from keras.layers import *
from keras.models import Model

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import h5py

# In[4]:

def train_model():
    model_resnet50_conv = ResNet50(weights='imagenet', include_top=False)
 

    #Create your own input format (here 3x200x200)
    input = Input(shape=(224,224,3),name = 'image_input')

    #Use the generated model 
    output_resnet50_conv = model_resnet50_conv(input)

    #Add the fully-connected layers 
    x = Flatten()(output_resnet50_conv)
    #x = Dropout(0.3)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)

    #Create your own model 
    model = Model(input=input, output=x)

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
   
    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[7]:

def preprocess_input(x, dim_ordering='default'):
    
    x=np.array(x,dtype=np.uint8)
        
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    # 'RGB'->'BGR'
    x = x[:, :, :, ::-1]
    return x


# In[9]:

def decode_predictions(preds):
    global CLASS_INDEX
    assert len(preds.shape) == 2 and preds.shape[1] == 10
    """
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    """
    indices = np.argmax(preds)
    results = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']    
    return results(indices)


# In[15]:

label = ''
frame = None
num_fold=1
kfold_weights_path = os.path.join('weights_kfold_vgg16_2.h5')
class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        # Load the VGG16 network
        print("[INFO] loading network...")
        self.model = train_model()
        self.model.load_weights(kfold_weights_path)
        while (~(frame is None)):
            #(inID, label) = self.predict(frame)
	    label = self.predict(frame)

    def predict(self, frame):
        #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float16)
        #image = image.transpose((2, 0, 1))
        #image = image.reshape((1,) + image.shape)
		
	X_test = []
	X_test.append(frame)
	test_data = np.array(X_test, dtype=np.uint8)
        test_data = test_data.astype('float16')   
        mean_pixel = [103.939, 116.779, 123.68]
        #print('Substract mean')
        test_data[:, :, :, 0] -= mean_pixel[0]   
        test_data[:, :, :, 1] -= mean_pixel[1]
        test_data[:, :, :, 2] -= mean_pixel[2]
        #test_data = preprocess_input(X_test)
	
        preds = self.model.predict(test_data)
	result=str(np.argmax(preds))
        return result

cap = cv2.VideoCapture(0)
if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

keras_thread = MyThread()
keras_thread.start()

while (True):
    ret, original = cap.read()

    frame = cv2.resize(original, (224, 224))

    # Display the predictions
    # print("ImageNet ID: {}, Label: {}".format(inID, label))
    cv2.putText(original, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()


# In[ ]:



