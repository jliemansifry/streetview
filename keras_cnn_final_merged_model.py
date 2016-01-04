import numpy as np
from keras.models import Sequential
import os
import time
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from scipy.misc import imread
import pandas as pd
from organizedImageData import write_filenames
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU

def load_data():
    df = pd.read_pickle("big_list_20000_with_categories.pkl")
    write_filenames(df, options = 'data_80x50_all')
    NESW = ['N', 'E', 'S', 'W']
    count = 19953
    all_X_data = np.zeros((count*4, 50, 80, 3))
    categories = list(df['county'].unique())
    categories.pop(29) # get rid of None
    category_name = 'county'
    all_y_data = np.zeros(count*4)
    print 'Loading data...'
    for df_idx in xrange(count):
        sub_idx = 0
        for cardinal_dir in NESW:
            image_name = (df.iloc[df_idx]['base_filename'] 
                          + cardinal_dir + '80x50.png')
            cnty = df.iloc[df_idx][category_name]
            if pd.isnull(cnty):
                continue
            else:
                image_class = categories.index(cnty)
            image_data = imread(image_name)
            idx_to_write = df_idx * 4 + sub_idx
            all_X_data[idx_to_write] = image_data
            all_y_data[idx_to_write] = image_class
            sub_idx += 1
    print len(df), len(all_X_data), len(all_y_data)
    return df, all_X_data, all_y_data, category_name, categories, count

def process_Xy(X, y, idx_offset, model_params):
    split_train, split_test = 72000, 7812
    X_train, X_test = X[split_test+idx_offset::4], X[idx_offset:split_test:4]
    y_train, y_test = y[split_test+idx_offset::4], y[idx_offset:split_test:4]
    X_train = X_train.reshape(X_train.shape[0], 3, model_params['img_rows'],
                                                   model_params['img_cols'])
    X_test = X_test.reshape(X_test.shape[0], 3, model_params['img_rows'], 
                                                model_params['img_cols'])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.
    y_train = np_utils.to_categorical(y_train, len(categories))
    y_test = np_utils.to_categorical(y_test, len(categories))
    return X_train, X_test, y_train, y_test

def save_model_param(model, unique_identifier, model_info, 
                     model_params, path_to_write):
    model_name = (category_name + '_' + str(model_params['batch_size'])
                  + '_batch_' + str(model_params['nb_epoch']) + '_epoch_' 
                  + str(count) + model_info + unique_identifier)
    print model_name
    print('Writing model ' + unique_identifier + ' info to file...')
    json_string = model.to_json()
    open(path_to_write + model_name 
         + '_model_arch.json', 'w').write(json_string)
    model.save_weights(path_to_write + model_name + '_model_weights.h5')

def run_model(X, y, category_name, categories, count):
    modelN_small_conv, model_params = add_model_params(categories)
    modelE_small_conv, model_params = add_model_params(categories)
    modelS_small_conv, model_params = add_model_params(categories)
    modelW_small_conv, model_params = add_model_params(categories)
    modelN_large_conv, model_params = add_model_params(categories, 2)
    modelE_large_conv, model_params = add_model_params(categories, 2)
    modelS_large_conv, model_params = add_model_params(categories, 2)
    modelW_large_conv, model_params = add_model_params(categories, 2)
    X_trainN, X_testN, y_trainN, y_testN = process_Xy(X, y, 0, model_params)
    X_trainE, X_testE, y_trainE, y_testE = process_Xy(X, y, 1, model_params)
    X_trainS, X_testS, y_trainS, y_testS = process_Xy(X, y, 2, model_params)
    X_trainW, X_testW, y_trainW, y_testW = process_Xy(X, y, 3, model_params)

    model = Sequential()
    model.add(Merge([modelN_small_conv, modelE_small_conv, 
                     modelS_small_conv, modelW_small_conv, 
                     modelN_large_conv, modelE_large_conv, 
                     modelS_large_conv, modelW_large_conv], mode = 'concat'))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha = 0.01))
    model.add(Dropout(0.5))
    model.add(Dense(len(categories)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    
    print '\nFitting model...\n'
    start = time.clock()
    model.fit([X_trainN, X_trainE, X_trainS, X_trainW, 
               X_trainN, X_trainE, X_trainS, X_trainW], y_trainN, 
               batch_size=model_params['batch_size'], 
               nb_epoch=model_params['nb_epoch'],
               show_accuracy=True, 
               verbose=1, 
               validation_data=([X_testN, X_testE, X_testS, X_testW,
                                 X_testN, X_testE, X_testS, X_testW], y_testN))
    stop = time.clock()
    total_run_time = (stop - start) / 60.
    print 'Done.\n'
    model_info = 'NESW_doubleconv_LeakyReLU_01'

    path_to_write = 'models/' + category_name + '/' + model_info + '/'
    if os.path.isdir('models/' + category_name) == False:
        os.makedirs('models/' + category_name + '/')
    if os.path.isdir(path_to_write[:-1]) == False:
        os.makedirs(path_to_write)

    save_model_param(model, 'merge', model_info, model_params, path_to_write)

    infoFile = path_to_write + model_info + '.txt'
    print('Writing model info to file...')
    f = open(infoFile, 'w')
    f.write('Run took {} minutes'.format(total_run_time))
    f.write('\n\nModel params:\n {}'.format('\n'.join(str(process) 
                                            for process 
                                            in sorted(model_params.iteritems()))))
    f.close()

def add_model_params(categories, conv_add = 0):
    model = Sequential()
    model_params = {'batch_size': 32,
                    'nb_epoch': 16,
                    'img_rows': 50,
                    'img_cols': 80,
                    'Convolution2D1': (256, 3 + conv_add, 3 + conv_add),
                    'Convolution2D2': (128, 3, 3),
                    'MaxPooling2D2': (2, 2),
                    'Convolution2D3': (128, 3, 3),
                    'MaxPooling2D3': (2, 2),
                    'Dropout3': 0.25,
                    'Flatten3': '',
                    'Dense3': 128}
    what_to_add = [Convolution2D(model_params['Convolution2D1'][0], 
                                 model_params['Convolution2D1'][1],
                                 model_params['Convolution2D1'][2],
                                 order_mode='valid',
                                 input_shape=(3, model_params['img_rows'], 
                                                 model_params['img_cols'])),
                    LeakyReLU(alpha = 0.01),
                    Convolution2D(model_params['Convolution2D2'][0],
                                  model_params['Convolution2D2'][1],
                                  model_params['Convolution2D2'][2]),
                    LeakyReLU(alpha = 0.01),
                    MaxPooling2D(pool_size=model_params['MaxPooling2D2']),
                    Convolution2D(model_params['Convolution2D3'][0],
                                  model_params['Convolution2D3'][1],
                                  model_params['Convolution2D3'][2]),
                    LeakyReLU(alpha = 0.01),
                    MaxPooling2D(pool_size=model_params['MaxPooling2D3']),
                    Dropout(model_params['Dropout3']),
                    Flatten(),
                    Dense(model_params['Dense3']),
                    LeakyReLU(alpha = 0.01)]
    for process in what_to_add:
        model.add(process)
    return model, model_params

if __name__ == '__main__':
    df, X, y, category_name, categories, count = load_data()
    run_model(X, y, category_name, categories, count)
