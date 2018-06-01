import plyvel
import pickle
import struct
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.models import Sequential 
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping  
from keras.callbacks import TensorBoard  

def createmodel(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (kernel_size, kernel_size), padding = 'valid', activation='relu', kernel_initializer='he_normal', input_shape=(128,501,1)))
    model.add(Conv2D(32, (kernel_size, kernel_size), padding = 'valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides = (2, 2)))
    model.add(Conv2D(32, (kernel_size, kernel_size), padding = 'valid', activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(32, (kernel_size, kernel_size), padding = 'valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides = (2, 2)))
    model.add(Conv2D(32, (kernel_size, kernel_size), padding = 'valid', activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(32, (kernel_size, kernel_size), padding = 'valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides = (2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer='he_normal'))
#    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))
    return(model)


if __name__ == '__main__':

    with open("dataset2.pic", "rb") as f:
        dataX, dataY = pickle.load(f)

    dataX, dataY = shuffle(dataX, dataY, random_state=0)

    trainX, testX = np.split(dataX, [6000])
    trainY, testY = np.split(dataY, [6000])

    batch_size = 50 # in each iteration, we consider 32 training examples at once
    num_epochs = 50 # we iterate 200 times over the entire training set
    kernel_size = 3 # we will use 5x5 kernels throughout
    pool_size = 2 # we will use 2x2 pooling throughout
    conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
    conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
    drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
    drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
    hidden_size = 4096 # the FC layer will have 512 neurons

    model = createmodel(2)
    model.compile(
              loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    tensorboard=TensorBoard(log_dir='./logs', write_graph=True)

    model.fit(trainX, trainY,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_split=0.1,
          callbacks=[tensorboard]
          )
    score = model.evaluate(testX, testY, batch_size=50, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model_json = model.to_json()
    json_file = open("eegmodel2.json", "w")
    json_file.write(model_json)
    json_file.close()
    model.save_weights("eegmodel2.h5")
