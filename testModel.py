import tensorflow as tf 
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
import cv2
import numpy as np

# issue: https://github.com/tensorflow/tensorflow/issues/24828
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


with open("trainingData2-50subjects.pickle", "rb") as f:
	x_train, y_train = pickle.load(f)

with open("trainingData2-39subjects.pickle", "rb") as f:
	x_train2, y_train2 = pickle.load(f)

with open("testingData2-20subjects.pickle", "rb") as f:
	x_test, y_test = pickle.load(f)

x_train = np.append(x_train, x_train2, axis=0)
y_train = np.append(y_train, y_train2, axis=0)

print(x_train.shape) 
print(y_train.shape)
print(x_test.shape)

'''
img_x = 128
img_y = 100
img_z = 3
input_shape = (img_y, img_x, img_z)
batch_size = 16
epochs = 10

model = Sequential()
model.add(Conv2D(8, kernel_size=(50,1), strides=(1,1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(16, (1,10), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1)
'''

model = load_model("saved_model.h5")
train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))

# if test_score[1] > 0.53:
# 	filepath = './saved_model.h5'
# 	save_model(model, filepath)