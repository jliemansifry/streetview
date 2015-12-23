from keras.models import Sequential
import theano
from keras.layers.core import Merge
from organizedImageData import write_filenames
import pandas as pd
import numpy as np
from scipy.misc import imread
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.optimizers import SGD

def load_data(category_name):
    df = pd.read_pickle("big_list_with_only_4_rock_classes_thru_14620.pkl")
    write_filenames(df, options = 'data_80x50')
    NESW = ['N', 'E', 'S', 'W']
    count = 50#df.shape[0]#6000 # just a test for now
    all_X_data = np.zeros((count*4, 50, 80, 3))
    if category_name == 'elev_gt_1800':
        categories = [False, True]
    elif category_name == 'county':
        categories = list(df['county'].unique())
    elif category_name == 'mcp':
        categories = 'mcp'
    #categories = ['0-5', '5-20', '20-250', '250-500', '500-3000']
    all_y_data = np.zeros(count*4)
    print 'Loading data...'
    for df_idx in xrange(count):
        sub_idx = 0
        for cardinal_dir in NESW:
            image_name = df.iloc[df_idx]['base_filename'] + cardinal_dir + '80x50.png'
            image_data = imread(image_name)
            #image_class = categories.index(df.iloc[df_idx]['rock_age'])
            if category_name == 'elev_gt_1800' or category_name == 'mcp':
                image_class = categories.index(df.iloc[df_idx][category_name])
            elif category_name == 'county':
                cnty = df.iloc[df_idx][category_name]
                if pd.isnull(cnty):
                    image_class = 65
                else:
                    image_class = categories.index(cnty)
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

## still will need to add Xy processing in order to extract all weights

def add_model_params(model, categories):
    model_params = {'batch_size': 64,
                    #'nb_classes': 2,
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
                    'Dense3': 64,
                    'Activation4': 'relu',
                    'Dropout4': 0.5}
                    # 'Dense4': len(categories),
                    # 'Activation5': 'softmax'}
    print model_params

    what_to_add = [Convolution2D(model_params['Convolution2D1'][0], # 0
                            model_params['Convolution2D1'][1],
                            model_params['Convolution2D1'][2],
                            border_mode='valid',
                            input_shape=(3, model_params['img_rows'], model_params['img_cols'])),
                  Activation(model_params['Activation1']), # 1
                  #MaxPooling2D(pool_size=model_params['MaxPooling2D1']),
                  Convolution2D(model_params['Convolution2D2'][0], # 2
                            model_params['Convolution2D2'][1],
                            model_params['Convolution2D2'][2]),
                  Activation(model_params['Activation2']), # 3
                  MaxPooling2D(pool_size=model_params['MaxPooling2D2']), # 4
                  #Dropout(model_params['Dropout2']),
                  Convolution2D(model_params['Convolution2D3'][0], # 5
                            model_params['Convolution2D3'][1],
                            model_params['Convolution2D3'][2]),
                  Activation(model_params['Activation3']), # 6
                  MaxPooling2D(pool_size=model_params['MaxPooling2D3']), # 7
                  Dropout(model_params['Dropout3']), # 8 
                  Flatten(), # 9
                  Dense(model_params['Dense3']), # 10
                  Activation(model_params['Activation4']), # 11
                  Dropout(model_params['Dropout4'])] # 12
                  # Dense(model_params['Dense4']), # 13
                  # Activation(model_params['Activation5'])] # 14
    for process in what_to_add:
        model.add(process)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model, what_to_add, model_params

def load_model(model_name):
    modelN = Sequential()
    modelE = Sequential()
    modelS = Sequential()
    modelW = Sequential()
    modelN, what_to_add, model_params = add_model_params(modelN, categories)
    modelE, what_to_add, model_params = add_model_params(modelE, categories)
    modelS, what_to_add, model_params = add_model_params(modelS, categories)
    modelW, what_to_add, model_params = add_model_params(modelW, categories)
    modelN = model_from_json(open(model_name + 'N_model_arch.json').read())
    modelE = model_from_json(open(model_name + 'E_model_arch.json').read())
    modelS = model_from_json(open(model_name + 'S_model_arch.json').read())
    modelW = model_from_json(open(model_name + 'W_model_arch.json').read())
    modelN.load_weights(model_name + 'N_model_weights.h5')
    modelE.load_weights(model_name + 'E_model_weights.h5')
    modelS.load_weights(model_name + 'S_model_weights.h5')
    modelW.load_weights(model_name + 'W_model_weights.h5')
    return modelN, modelE, modelS, modelW

def concat_models(*args):
    models = [model_dir for model_dir in args]
    model = Sequential()
    model.add(Merge(models, mode = 'concat'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(categories)))
    model.add(Activation('softmax'))
    how_many_models = len(models)
    # model = Sequential()
    # model.add(Merge(models, mode = 'concat', concat_axis = 1))
    # model.add(Dense(10))
    # model.add(Activation('softmax'))
    return model, how_many_models

def run_concat_model(model, how_many_models, X, y):
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.fit([X] * how_many_models, y, batch_size=128, nb_epoch=20)
    return model

def get_activations(model, layer, X_batch, NESW = False):
    if NESW == True:
        get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    else:
        get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

def predict(X, model):
    classes = model.predict_classes(X, batch_size=32)
    proba = model.predict_proba(X, batch_size=32)
    #cla = model.predict_classes(X, batch_size=32)
    pro = model._predict(X, batch_size=32)
    return classes, proba, pro

def write_to_df(proba, categories, write_to_df = None, df = None):
    reshaped_probas = proba.reshape((proba.shape[0]/4., 4*proba.shape[1]))
    if df is None:
        df = pd.DataFrame(reshaped_probas)
    else:
        pass

if __name__ == '__main__':
    df, X, y, category_name, categories, count = load_data('county')
    #model = load_model('models/elev_gt_1800_64_batch_16_epoch_14621_locations_256x3x3_128x3x3_128x3x3_conv_2x2__layers23pool_seedset_model_')
    modelN, modelE, modelS, modelW = load_model('models/county/_NESW_dense256_relu_drop05_dense3_/county_64_batch_16_epoch_14621_NESW_dense256_relu_drop05_dense3_')
    model, how_many_models = concat_models(modelN, modelE, modelS, modelW)

    #classes, proba, pro = predict(X, model)
