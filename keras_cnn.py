import numpy as np
np.random.seed(1337)
from keras.models import Sequential
import itertools
import pydot
import os
from keras.utils import visualize_util# import plot
import time
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
# from imageAnalysisFunctions import cv2_image
from scipy.misc import imread
import pandas as pd
from organizedImageData import write_filenames
from keras.utils import np_utils

def load_data():
    df = pd.read_pickle("big_list_with_only_4_rock_classes_thru_14620.pkl")
    #df = pd.read_pickle("big_list_with_classes_thru_14620.pkl")
   #write_filenames(df, options = 'data_160x100_thru_14243/data_160x100')
    write_filenames(df, options = 'data_80x50')
    NESW = ['N', 'E', 'S', 'W']
    count = df.shape[0]#6000 # just a test for now
    all_X_data = np.zeros((count*4, 50, 80, 3))
    categories = [True, False]
    category_name = 'elev_gt_1800'
    #categories = ['0-5', '5-20', '20-250', '250-500', '500-3000']
    all_y_data = np.zeros(count*4)
    print 'Loading data...'
    for df_idx in xrange(count):
        sub_idx = 0
        for cardinal_dir in NESW:
            image_name = df.iloc[df_idx]['base_filename'] + cardinal_dir + '80x50.png'
            image_data = imread(image_name)
            #image_class = categories.index(df.iloc[df_idx]['rock_age'])
            image_class = categories.index(df.iloc[df_idx][category_name])
            idx_to_write = df_idx * 4 + sub_idx
            all_X_data[idx_to_write] = image_data
            all_y_data[idx_to_write] = image_class
            sub_idx += 1
    return df, all_X_data, all_y_data, category_name, categories, count

def run_model(X, y, category_name, categories, count):
    model = Sequential()
    model, what_to_add, model_params = add_model_params(model, categories)

    split_train, split_test = len(X) * 0.9, len(X) * 0.1
    X_train, X_test = X[:split_train], X[split_train:len(X)]
    y_train, y_test = y[:split_train], y[split_train:len(X)]

    X_train = X_train.reshape(X_train.shape[0], 3, model_params['img_rows'], model_params['img_cols'])
    X_test = X_test.reshape(X_test.shape[0], 3, model_params['img_rows'], model_params['img_cols'])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, len(categories))
    Y_test = np_utils.to_categorical(y_test, len(categories))
    print '\nFitting model...\n'
    start = time.clock()
    model.fit(X_train, Y_train, 
        batch_size=model_params['batch_size'], 
        nb_epoch=model_params['nb_epoch'],
                show_accuracy=True, 
        verbose=1, 
        validation_data=(X_test, Y_test))
    stop = time.clock()
    total_run_time = (stop - start) / 60.
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
    print('Test score:', score)
    print 'Done.\n'

    model_name = category_name + '_' + str(model_params['batch_size']) + '_batch_' + str(model_params['nb_epoch']) + '_epoch_' + str(count) + '_locations_256x3x3_256x3x3_conv_2x2_pool'
    print model_name
    if os.path.isdir('models/' + category_name) == False:
        os.makedirs('models/' + category_name + '/')
    path = 'models/' + category_name + '/'

    print('Writing model info to file...')
    infoFile = path + model_name + '.txt'
    f = open(infoFile, 'w')
    f.write('Run took {} minutes'.format(total_run_time))
    f.write('\nX_train shape: {}'.format(X_train.shape))
    f.write('\nX_train number of samples: {}'.format(X_train.shape[0]))
    f.write('\nX_test number of samples: {}'.format(X_test.shape[0]))
    f.write('\n\nModel params:\n {}'.format('\n'.join(str(process) for process in sorted(model_params.iteritems()))))
    f.write('\nModel evaluation on test data from Keras: {}'.format(score))
    f.close()
    #plot(model, path + model_name + '_graph.ps', model_params)
    json_string = model.to_json()
    open(path + model_name + '_model_arch.json', 'w').write(json_string)
    model.save_weights(path + model_name + '_model_weights.h5')

def add_model_params(model, categories):
    model_params = {'batch_size': 64,
                    'nb_classes': 2,
                    'nb_epoch': 16,
                    'img_rows': 50,
                    'img_cols': 80,
                    'Convolution2D1': (256, 3, 3),
                    'Activation1': 'relu', 
                    'MaxPooling2D1': (2, 2),
                    'Convolution2D2': (256, 3, 3),
                    'Activation2': 'relu',
                    'MaxPooling2D2': (2, 2),
                    #'Convolution2D3': (32, 5, 5),
                    #'Activation3': 'relu',
                    #'MaxPooling2D3': (2, 2),
                    'Dropout2': 0.25, 
                    'Flatten2': '',
                    'Dense2': 16,
                    'Activation3': 'relu', 
                    'Dropout3': 0.5, 
                    'Dense3': len(categories),
                    'Activation4': 'softmax'}
    print model_params
    what_to_add = [Convolution2D(model_params['Convolution2D1'][0], 
                    model_params['Convolution2D1'][1],
                    model_params['Convolution2D1'][2],
                    border_mode='valid',
                    input_shape=(3, model_params['img_rows'], model_params['img_cols'])),
                    Activation(model_params['Activation1']),
                    MaxPooling2D(pool_size=model_params['MaxPooling2D1']),
                    Convolution2D(model_params['Convolution2D2'][0],
                            model_params['Convolution2D2'][1],
                            model_params['Convolution2D2'][2]),
                    Activation(model_params['Activation2']),
                    MaxPooling2D(pool_size=model_params['MaxPooling2D2']),
                    #Convolution2D(model_params['Convolution2D3'][0],
                    #        model_params['Convolution2D3'][1],
                    #        model_params['Convolution2D3'][2]),
                    #  Activation(model_params['Activation3']),
                    #  MaxPooling2D(pool_size=model_params['MaxPooling2D3']),
                    Dropout(model_params['Dropout2']),
                    Flatten(),
                    Dense(model_params['Dense2']),
                    Activation(model_params['Activation3']),
                    Dropout(model_params['Dropout3']),
                    Dense(model_params['Dense3']),
                    Activation(model_params['Activation4'])]
    for process in what_to_add:
        model.add(process)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model, what_to_add, model_params

def plot(model, to_file, model_params):
    graph = pydot.Dot(graph_type='digraph')
    if type(model) == Sequential:
        previous_node = None
        written_nodes = []
        n = 1
        for node in model.get_config()['layers']:
            # append number in case layers have same name to differentiate
            if (node['name'] + str(n)) in written_nodes:
                n += 1
        node_name = node['name'] + str(n)
        #print node_name
        #print model_params[node_name]
            current_node = pydot.Node(node_name + '--' + str(model_params[node_name]))
            written_nodes.append(node_name)
            graph.add_node(current_node)
            if previous_node:
                graph.add_edge(pydot.Edge(previous_node, current_node))
            previous_node = current_node
        graph.write_png(to_file)

if __name__ == '__main__':
    df, X, y, category_name, categories, count = load_data()
    run_model(X, y, category_name, categories, count)
