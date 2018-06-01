import plyvel
import pickle
import struct
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential 
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.callbacks import EarlyStopping  

def datagenerator(db, dX, dY):
    for key, value in db:
        data = pickle.loads(value)
        dtx = preprocessing.normalize(data[0])
        dty = data[1]
        print(data[2])
        dtx.shape = (1, 128, 1001, 1)
        dty.shape = (1, 2)
        dX = np.vstack((dX, dtx))
        dY = np.vstack((dY, dty))
        del(dtx)
        del(dty)
        del(data)
    return dX, dY


if __name__ == '__main__':

    db = plyvel.DB('../eegdb5_10/', create_if_missing=False)
    dX = np.zeros(shape=(0, 128, 1001, 1), dtype = np.float32)
    dY = np.zeros(shape = (0, 2), dtype = np.int32)
    dataX, dataY = datagenerator(db, dX, dY)
    print(dataX.shape)
    print(dataY.shape)
    db.close()
    with open ("dataset5_10.pic", "wb") as f:
        pickle.dump((dataX, dataY), f)
