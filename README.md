# CNN Model using keras functional API
Application of keras functional API for custom CNN model development. 


* Model Structure
```python
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
model.fit(train, steps_per_epoch=100, validation_data=valid, validation_steps=50, epochs=12,verbose=2)
```
