from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
# from imageAnalysisFunctions import cv2_image
from scipy.misc import imread
import pandas as pd
from organizedImageData import write_filenames
import numpy as np
from keras.utils import np_utils

def load_data():
    df = pd.read_pickle("big_list_with_all_classes.pkl")
    write_filenames(df, options = 'data_160x100')
    NESW = ['N', 'E', 'S', 'W']
    count = 6000 # just a test for now
    all_X_data = np.zeros((count*4, 100, 160, 3))
    categories = ['0-5', '5-20', '20-250', '250-500', '500-3000']
    all_y_data = np.zeros(6000*4)
    for df_idx in xrange(6000):
        sub_idx = 0
        for cardinal_dir in NESW:
            print df_idx
            image_name = df.iloc[df_idx]['base_filename'] + cardinal_dir + '160x100.png'
            image_data = imread(image_name)
            image_class = categories.index(df.iloc[df_idx]['rock_age'])
            idx_to_write = df_idx * 4 + sub_idx
            all_X_data[idx_to_write] = image_data
            all_y_data[idx_to_write] = image_class
            sub_idx += 1
    return df, all_X_data, all_y_data

def run_model(X, y):
    X_train, X_test = X[:20000], X[20000:]
    y_train, y_test = y[:20000], y[20000:]
    
    batch_size = 128
    nb_classes = 5
    nb_epoch = 12

    # input image dimensions
    img_rows, img_cols = 100, 160
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # the data, shuffled and split between tran and test sets
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(3, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
            show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])

if __name__ == '__main__':
    df, X, y = load_data()
    run_model(X, y)
