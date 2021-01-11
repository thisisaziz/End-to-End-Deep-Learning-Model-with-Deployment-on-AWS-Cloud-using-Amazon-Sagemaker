#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[15]:





# In[15]:





# In[15]:





# In[15]:





# In[ ]:





# In[8]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
import argparse
import os
import re
import time


# In[2]:


def model():

    from keras.applications import InceptionResNetV2
    model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(220, 220, 3))

    for layer in model.layers:
            layer.trainable = False

    from keras.layers import Dense, GlobalAveragePooling2D, Dropout,Flatten
    from keras.models import Model, Sequential

    # Although this part can be done also with the functional API, I found that for this simple models, this becomes more intuitive
    transfer_model=model.output
    transfer_model=(GlobalAveragePooling2D())(transfer_model)
    #transfer_model = Flatten(name="flatten")(transfer_model)
    transfer_model=(Dense(1024,activation='relu'))(transfer_model)
    transfer_model=(Dropout(0.4))(transfer_model)
    transfer_model=(Dense(50, activation="softmax"))(transfer_model) # Finally our activation layer! we use 10 outputs as we have 10 monkey species (labels)
    modelnew= Model(inputs=model.input,outputs= transfer_model)
    return modelnew


# In[3]:





# In[6]:


def preprocess(image_path,batch_size):

    inp_size=(220,220)

    train_data_dir=image_path
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=inp_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='training') # set as training data

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir, # same directory as training data
        target_size=inp_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='validation')
    return train_generatorgener,validation_generator

def main(args):
    epochs       = args.epochs
    lr           = args.learning_rate
    batch_size   = args.batch_size
    momentum     = args.momentum
    weight_decay = args.weight_decay
    optimizer    = args.optimizer

    # SageMaker options
    training_dir   = args.training
    validation_dir = args.validation
    eval_dir       = args.eval

    train_dataset = preprocess(training_dir+'/train.tfrecords',  batch_size)
    val_dataset   = preprocess(training_dir+'/train.tfrecords', batch_size)
    eval_dataset  = preprocess(eval_dir+'/test.tfrecords', batch_size)
    
    input_shape = (220, 220, 3)
    model = model()
    
    # Optimizer
    if optimizer.lower() == 'sgd':
        opt = SGD(lr=lr, decay=weight_decay, momentum=momentum)
    else:
        opt = Adam(lr=lr, decay=weight_decay)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)    
    model.compile(loss = 'categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy'])
    modelnew.fit(
    train_dataset,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = val_dataset,
    validation_steps = nb_validation_samples // batch_size)


# In[7]:


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument('--epochs',        type=int,   default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size',    type=int,   default=32)
    parser.add_argument('--weight-decay',  type=float, default=2e-4)
    parser.add_argument('--momentum',      type=float, default='0.9')
    parser.add_argument('--optimizer',     type=str,   default='sgd')

    # SageMaker parameters
    parser.add_argument('--training',         type=str,   default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])
    
    args = parser.parse_args()
    main(args)


# In[ ]:




