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
    '''
    INPUT:  None
    OUTPUT: (1) Pandas dataframe: the raw df loaded from .pkl
            (2) 4D numpy array: All image data loaded into a huge tensor
                with dimensionality (4*count, 50, 80, 3)
                Note: this will need to be reshaped for Keras to be happy
            (3) 1D numpy array: All y data, as indices of classes created
                by df['county'].unique()
            (4) string: the category name
            (5) list: the list used to index the target
            (6) integer: the count of df indicies used

    This function will load all the image data and target values into memory.
    It is set up to use 'county' as the target but would require minimal edits
    to use different class labels. If the image data tensor ever became too
    large to fit in memory, it would be necessary to load it in smaller
    batches that could then be trained on.
    '''
    df = pd.read_pickle("big_list_20000_with_categories.pkl")
    write_filenames(df, options='data_80x50_all')
    NESW = ['N', 'E', 'S', 'W']
    unique_count = 19953
    unique_directions = 4
    total_num_images = unique_count * unique_directions
    all_X_data = np.zeros((total_num_images, 50, 80, 3))
    categories = df['county'].unique()
    null_categories = np.where(pd.isnull(categories))[0]
    category_list = list(categories)
    category_list.pop(null_categories)
    category_name = 'county'
    all_y_data = np.zeros(total_num_images)
    print 'Loading data...'
    for df_idx in xrange(unique_count):
        sub_idx = 0
        for cardinal_dir in NESW:
            image_name = (df.iloc[df_idx]['base_filename'] +
                          cardinal_dir + '80x50.png')
            cnty = df.iloc[df_idx][category_name]
            if pd.isnull(cnty):
                continue
            else:
                image_class = categories.index(cnty)
            image_data = imread(image_name)
            idx_to_write = df_idx * unique_directions + sub_idx
            all_X_data[idx_to_write] = image_data
            all_y_data[idx_to_write] = image_class
            sub_idx += 1
    return df, all_X_data, all_y_data, category_name, categories, unique_count


def process_Xy(X, y, idx_offset, model_params):
    '''
    INPUT:  (1) 4D numpy array: all X
            (2) 1D numpy array: all y
            (3) integer: idx offset to be used when reading through the X
                tensor. Images are loaded such that index 0 is facing North,
                1: East, 2: South, 3: West, 4: North, etc.
                This allows the N, E, S, and W image data to be split back
                out such that unique X will be used for each
                cardinal direction.
            (4) dictionary of model params for ease of processing X
    OUTPUT: (1) 4D numpy array: X train split
            (2) 4D numpy array: X test split
            (3) 1D numpy array: y train split
            (4) 1D numpy array: y test split

    Because a fractional split of len(data) results in a float, the
    train/test split is hardcoded in.
    '''
    split_train, split_test = 72000, 7812
    X_train, X_test = X[split_test+idx_offset::4], X[idx_offset:split_test:4]
    y_train, y_test = y[split_test+idx_offset::4], y[idx_offset:split_test:4]
    X_train = X_train.reshape(X_train.shape[0], 3,
                              model_params['img_rows'],
                              model_params['img_cols'])
    X_test = X_test.reshape(X_test.shape[0], 3,
                            model_params['img_rows'],
                            model_params['img_cols'])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.
    y_train = np_utils.to_categorical(y_train, len(categories))
    y_test = np_utils.to_categorical(y_test, len(categories))
    return X_train, X_test, y_train, y_test


def save_model_param(model, model_info, model_params,
                     path_to_write, category_name, unique_count):
    '''
    INPUT:  (1) Keras model object
            (2) string: info about model for the filename
            (3) dictionary of model parameters for filename
            (4) string: path to save the model to
            (5) string: category name for filename
            (6) integer: unique count of locations for filename
    '''
    model_name = '''{}_{}_batch_{}_epoch_{}{}'''.format(
                 category_name,
                 model_params['batch_size'],
                 model_params['nb_epoch'],
                 unique_count,
                 model_info)
    print model_name
    print('Writing model info to file...')
    json_string = model.to_json()
    open(path_to_write + model_name +
         '_model_arch.json', 'w').write(json_string)
    model.save_weights(path_to_write + model_name + '_model_weights.h5')


def run_model(X, y, category_name, categories, unique_count):
    '''
    INPUT:  (1) 4D numpy array of all X data
            (2) 1D numpy array of target
            (3) string: category name
            (4) list of categories
            (5) integer: count of unique locations
    OUTPUT: None

    This function will build the model and run it. It creates the structure
    of distinct N, E, S, and W models with different size convolutions, merges
    them, and adds structure to the merged model. It feeds the distinct
    X data for each cardinal direction to the merged model, which is then
    compiled, trained, and saved.
    Tuning of the merged model parameters happens here.
    '''
    modelN_small_conv, model_params = add_model_params()
    modelE_small_conv, model_params = add_model_params()
    modelS_small_conv, model_params = add_model_params()
    modelW_small_conv, model_params = add_model_params()
    modelN_large_conv, model_params = add_model_params(conv_add=2)
    modelE_large_conv, model_params = add_model_params(conv_add=2)
    modelS_large_conv, model_params = add_model_params(conv_add=2)
    modelW_large_conv, model_params = add_model_params(conv_add=2)
    X_trainN, X_testN, y_trainN, y_testN = process_Xy(X, y, 0, model_params)
    X_trainE, X_testE, y_trainE, y_testE = process_Xy(X, y, 1, model_params)
    X_trainS, X_testS, y_trainS, y_testS = process_Xy(X, y, 2, model_params)
    X_trainW, X_testW, y_trainW, y_testW = process_Xy(X, y, 3, model_params)

    model = Sequential()
    model.add(Merge([modelN_small_conv, modelE_small_conv,
                     modelS_small_conv, modelW_small_conv,
                     modelN_large_conv, modelE_large_conv,
                     modelS_large_conv, modelW_large_conv], mode='concat'))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.01))
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

    save_model_param(model, 'merge', model_info, model_params,
                     path_to_write, category_name, unique_count)

    infoFile = path_to_write + model_info + '.txt'
    print('Writing model info to file...')
    f = open(infoFile, 'w')
    f.write('Run took {} minutes'.format(total_run_time))
    processes = sorted(model_params.iteritems())
    f.write('\n\nModel params:\n {}'.format('\n'.join(str(process)
                                            for process
                                            in processes)))
    f.close()


def add_model_params(conv_add=0):
    '''
    INPUT:  (1) integer: the amount to increase the base convolution size by
                in pixels for the first layer.
    OUTPUT: (1) Keras model object, with all structure added (but uncompiled)
            (2) dictionary of model parameters that were added

    This function initializes the model and adds the specified structure.
    Tuning any of the base model parameters happens here.
    '''
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
                                 input_shape=(3,
                                              model_params['img_rows'],
                                              model_params['img_cols'])),
                   LeakyReLU(alpha=0.01),
                   Convolution2D(model_params['Convolution2D2'][0],
                                 model_params['Convolution2D2'][1],
                                 model_params['Convolution2D2'][2]),
                   LeakyReLU(alpha=0.01),
                   MaxPooling2D(pool_size=model_params['MaxPooling2D2']),
                   Convolution2D(model_params['Convolution2D3'][0],
                                 model_params['Convolution2D3'][1],
                                 model_params['Convolution2D3'][2]),
                   LeakyReLU(alpha=0.01),
                   MaxPooling2D(pool_size=model_params['MaxPooling2D3']),
                   Dropout(model_params['Dropout3']),
                   Flatten(),
                   Dense(model_params['Dense3']),
                   LeakyReLU(alpha=0.01)]
    for process in what_to_add:
        model.add(process)
    return model, model_params


if __name__ == '__main__':
    df, X, y, category_name, categories, unique_count = load_data()
    run_model(X, y, category_name, categories, unique_count)
