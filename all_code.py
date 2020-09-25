
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random 
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility

s12=1472
tf.random.set_seed(s12)
np.random.seed(s12)
random.seed(s12)

# Data Pre-Processing
path1='./datasets/train/'
path2='./datasets/valid/'
train=ImageDataGenerator(rescale=1.0/255.0,
                         rotation_range=2,
                         fill_mode="nearest",
                         height_shift_range=1.01,
                         shear_range=0.01,
                         zoom_range=[1,1.01],
                         horizontal_flip=False).flow_from_directory(path1, 
                                                                    color_mode="grayscale",
                                                                    target_size=(64,64),
                                                                    batch_size=48,
                                                                    shuffle=True)                 
valid=ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(path2, 
                                                                color_mode="grayscale",
                                                                    target_size=(64,64),
                                                                batch_size=24,
                                                                shuffle=True)

# Model Setup
in0 = tf.keras.Input(shape=(64,64,1))

x11 = tf.keras.layers.Conv2D(filters=16,kernel_size=(2,2),padding="same", activation=tf.nn.relu)(in0)
x12 = tf.keras.layers.Conv2D(filters=32,kernel_size=(2,2),padding="same", activation=tf.nn.relu)(x11)
x13 = tf.keras.layers.Conv2D(filters=64,kernel_size=(2,2),padding="same", activation=tf.nn.relu)(x12)
out1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x13)

x21 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding="same", activation=tf.nn.relu)(in0)
x22 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding="same", activation=tf.nn.relu)(x21)
out2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x22)

x31 = tf.keras.layers.Conv2D(filters=5,kernel_size=(4,4),padding="same", activation=tf.nn.relu)(in0)
out3 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x31)

x41 = tf.keras.layers.Conv2D(filters=32,kernel_size=(2,2),padding="same", activation=tf.nn.relu)(in0)
x42 = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same", activation=tf.nn.relu)(x41)
x43 = tf.keras.layers.Conv2D(filters=8,kernel_size=(4,4),padding="same", activation=tf.nn.relu)(x42)
out4 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x43)

x1 = tf.keras.layers.Concatenate()([out1,out2,out3,out4]) 
x2 = tf.keras.layers.Conv2D(filters=32,kernel_size=(2,2),padding="same", activation=tf.nn.relu)(x1)
x3 = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same", activation=tf.nn.relu)(x2)
x4 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x3)
x5 =tf.keras.layers.Flatten()(x4)
x6 =tf.keras.layers.Dropout(0.2)(x5)
x7 = tf.keras.layers.Dense(72, activation=tf.nn.relu)(x6)
out0 = tf.keras.layers.Dense(6, activation=tf.nn.softmax)(x7)
model = tf.keras.Model(inputs=in0, outputs=out0)

model.summary()
from tensorflow.keras.optimizers import Adam
model.compile(tf.keras.optimizers.SGD(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

# Visualize
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='my_model0124.png', show_shapes=True, show_layer_names=True)

# Training
hist=model.fit(train, steps_per_epoch=100, validation_data=valid, validation_steps=50, epochs=12,verbose=2)

# Results
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('history: the CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('history: the CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
