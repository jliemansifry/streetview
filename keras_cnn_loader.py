from keras.models import Sequential
from organizedImageData import write_filenames
import pandas as pd
import numpy as np
from scipy.misc import imread
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.optimizers import SGD
def load_data():
    df = pd.read_pickle("big_list_with_only_4_rock_classes_thru_14620.pkl")
    write_filenames(df, options = 'data_80x50')
    NESW = ['N', 'E', 'S', 'W']
    count = 50#df.shape[0]#6000 # just a test for now
    all_X_data = np.zeros((count*4, 50, 80, 3))
    categories = [False, True]
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
    X = all_X_data.reshape(all_X_data.shape[0], 3, 50, 80)
    #X_test = all_X_data.reshape(X_test.shape[0], 3, model_params['img_rows'], model_params['img_cols'])
    X = X.astype('float32')
    #X_test = X_test.astype('float32')
    X /= 255.
    #X_test /= 255.
    return df, X, all_y_data, category_name, categories, count

def add_model_params(model,categories):
    model_params = {'batch_size': 64,
                    'nb_classes': 2,
                    'nb_epoch': 16,
                    'img_rows': 50,
                    'img_cols': 80,
                    'Convolution2D1': (256, 3, 3),
                    'Activation1': 'relu',
                    #'MaxPooling2D1': (2, 2),
                    'Convolution2D2': (128, 3, 3),
                    'Activation2': 'relu',
                    'MaxPooling2D2': (2, 2),
                    #'Dropout2': 0.25,
                    'Convolution2D3': (128, 3, 3),
                    'Activation3': 'relu',
                    'MaxPooling2D3': (2, 2),
                    'Dropout3': 0.25,
                    'Flatten3': '',
                    'Dense3': 16,
                    'Activation4': 'relu',
                    'Dropout4': 0.5,
                    'Dense4': len(categories),
                    'Activation5': 'softmax'}
    print model_params
    what_to_add = [Convolution2D(model_params['Convolution2D1'][0],
                            model_params['Convolution2D1'][1],
                            model_params['Convolution2D1'][2],
                            border_mode='valid',
                            input_shape=(3, model_params['img_rows'], model_params['img_cols'])),
                  Activation(model_params['Activation1']),
                  #MaxPooling2D(pool_size=model_params['MaxPooling2D1']),
                  Convolution2D(model_params['Convolution2D2'][0],
                            model_params['Convolution2D2'][1],
                            model_params['Convolution2D2'][2]),
                  Activation(model_params['Activation2']),
                  MaxPooling2D(pool_size=model_params['MaxPooling2D2']),
                  #Dropout(model_params['Dropout2']),
                  Convolution2D(model_params['Convolution2D3'][0],
                            model_params['Convolution2D3'][1],
                            model_params['Convolution2D3'][2]),
                  Activation(model_params['Activation3']),
                  MaxPooling2D(pool_size=model_params['MaxPooling2D3']),
                  Dropout(model_params['Dropout3']),
                  Flatten(),
                  Dense(model_params['Dense3']),
                  Activation(model_params['Activation4']),
                  Dropout(model_params['Dropout4']),
                  Dense(model_params['Dense4']),
                  Activation(model_params['Activation5'])]
    for process in what_to_add:
        model.add(process)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model, what_to_add, model_params

def load_model(model_name):
    model = Sequential()
    model, what_to_add, model_params = add_model_params(model, categories)
    model = model_from_json(open(model_name + 'arch.json').read())
    model.load_weights(model_name + 'weights.h5')
    return model

def predict(X, model):
    classes = model.predict_classes(X, batch_size=32)
    proba = model.predict_proba(X, batch_size=32)
    return classes, proba

if __name__ == '__main__':
    df, X, y, category_name, categories, count = load_data()
    #model = load_model('models/elev_gt_1800_64_batch_16_epoch_14621_locations_256x3x3_128x3x3_128x3x3_conv_2x2__layers23pool_seedset_model_arch')
    #model = load_model('models/elev_gt_1800_64_batch_16_epoch_14621_locations_256x3x3_128x3x3_128x3x3_conv_2x2__layers23pool_seedset_model_weights.h5')
    model = load_model('models/elev_gt_1800_64_batch_16_epoch_14621_locations_256x3x3_128x3x3_128x3x3_conv_2x2__layers23pool_seedset_model_')
    classes, proba = predict(X, model)
