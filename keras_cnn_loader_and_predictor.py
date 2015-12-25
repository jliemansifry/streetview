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
    ''' 
    INPUT:  (1) string: The category name that is being predicted
    OUTPUT: (1) df: The full Pandas DataFrame that was loaded and used
            (2) 4D numpy array: All X data (# of images x RGB x height x width)
            (3) 1D numpy array: All y data (target category indices)
            (4) list: corresponding category names to (3)
    '''
    df = pd.read_pickle("big_list_with_only_4_rock_classes_thru_14620.pkl")
    write_filenames(df, options = 'data_80x50')
    NESW = ['N', 'E', 'S', 'W']
    count = df.shape[0]
    all_X_data = np.zeros((count*4, 50, 80, 3))
    if category_name == 'elev_gt_1800':
        categories = [False, True]
    elif category_name == 'county':
        categories = list(df['county'].unique())
    elif category_name == 'mcp':
        categories = 'mcp'
    elif category_name == 'rock_age':
        categories = ['0-5', '5-20', '20-250', '250-3000']
    all_y_data = np.zeros(count*4)
    print 'Loading data...'
    for df_idx in xrange(count):
        sub_idx = 0
        for cardinal_dir in NESW:
            image_name = df.iloc[df_idx]['base_filename'] + cardinal_dir + '80x50.png'
            image_data = imread(image_name)
            if category_name == 'elev_gt_1800' or category_name == 'mcp' or category_name == 'rock_age':
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
    X = X.astype('float32')
    X /= 255.
    return df, X, all_y_data, categories

def load_models(model_name):
    ''' 
    INPUT:  (1) string: the full path to the model weights and architecture
    OUTPUT: (1) The N, E, S, and W models with 
                trained weights and architecture

    Four Sequential model objects are created. The appropriate model 
    structure is read from a json file created during training, 
    then the trained weights are loaded onto this structure.
    All four models are returned.
    '''    
    modelN = model_from_json(open(model_name + 'N_model_arch.json').read())
    modelE = model_from_json(open(model_name + 'E_model_arch.json').read())
    modelS = model_from_json(open(model_name + 'S_model_arch.json').read())
    modelW = model_from_json(open(model_name + 'W_model_arch.json').read())
    modelN.load_weights(model_name + 'N_model_weights.h5')
    modelE.load_weights(model_name + 'E_model_weights.h5')
    modelS.load_weights(model_name + 'S_model_weights.h5')
    modelW.load_weights(model_name + 'W_model_weights.h5')
    return modelN, modelE, modelS, modelW

def get_activations(model, layer, X_batch):
    ''' 
    INPUT:  (1) Keras Sequential model object
            (2) integer: The layer to extract weights from
            (3) 4D numpy array: All the X data you wish to extract 
                activations for
    OUTPUT: (1) numpy array: Activations for that layer
    '''
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch)
    return activations

def get_merged_activations(modelN, modelE, modelS, modelW, X):
    ''' 
    INPUT:  (1) Keras Sequential model object: North
            (2) Keras Sequential model object: East
            (3) Keras Sequential model object: South
            (4) Keras Sequential model object: West
            (5) 4D numpy array: All the X data
    OUTPUT: (1) numpy array: vertically stacked activations for the 4 models
    
    The activations for the final layer of each model are calcualted,
    then vertically stacked. These are the activations that are feeding
    into the merged model. Not actually used, but good to have a grasp of. 
    '''
    N_activations = get_activations(modelN, 12, X[::4])
    E_activations = get_activations(modelE, 12, X[1::4])
    S_activations = get_activations(modelS, 12, X[2::4])
    W_activations = get_activations(modelW, 12, X[3::4])
    final_layer = np.vstack((N_activations, E_activations, S_activations, W_activations))
    return final_layer

def build_merged_model(model_name):
    ''' 
    INPUT:  (1) string: the full path to the model weights and architechure
    OUTPUT: (1) Merged keras model
    '''
    model = model_from_json(open(model_name + 'merge_model_arch.json').read())
    model.load_weights(model_name + 'merge_model_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model

def test_equality_of_build_methods(model, modelN):
    ''' 
    INPUT:  (1) NESW merged model, loaded from json and with trained weights
            (2) N model, loaded from json and with trained weights
    OUTPUT: None

    This is a test to show that the process of building the merged model
    from the N, E, S, and W models separately is equivalent to loading the 
    merged model (NESW) from json with the trained weights. 

    In other words, Keras makes rebuilding a model superbly easy, even
    if this model is a merge of other models. It will keep track of all 
    the structure and the trained weights for you.
    '''
    merged_layer = model.layers[0]
    merged_layer_weights = merged_layer.get_weights()
    N_weights = modelN.get_weights()
    print 'It is {} that the weights are the same'.format(all(
        merged_layer_weights[7] == N_weights[7]))

def return_specified_proba(X, idx, model_name, categories, NESW_merged = None):
    ''' 
    INPUT:  (1) 4D numpy array: All X data
            (2) integer: index to determine probabilities for
            (3) string: the full path to the model name
            (4) list: all categories
            (5) optional previously loaded model 
    OUTPUT: (1) dict: categories and their associated probabilities

    Return the probabilities associated with each category that the model
    has been trained on for a specific location in the dataset. 
    If no model has been specified, return the model as well so that it does 
    not need to be reloaded in the future.
    '''
    model_built = False
    if NESW_merged is None:
        model_built = True
        NESW_merged = build_merged_model(model_name)
    N_idx = idx * 4; E_idx = idx * 4 + 1
    S_idx = idx * 4 + 2; W_idx = idx * 4 + 3
    end_idx = idx * 4 + 4
    final_probas = NESW_merged.predict_proba([X[N_idx:end_idx:4], X[E_idx:end_idx:4], X[S_idx:end_idx:4], X[W_idx:end_idx:4]], batch_size = 1)[0]
    probas_dict = {c: p for c, p in zip(categories, final_probas)}
    if model_built:
        return probas_dict, NESW_merged
    return probas_dict

if __name__ == '__main__':
    df, X, y, categories = load_data('county')
    model_name = 'models/county/_NESW_dense256_relu_drop05_dense3_/county_64_batch_16_epoch_14621_NESW_dense256_relu_drop05_dense3_'
    # modelN, modelE, modelS, modelW = load_models(model_name)
    #X_merged = get_merged_activations(modelN, modelE, modelS, modelW, X)
    #NESW = build_merged_model_as_standalone(X_merged, model_name, categories)
    #final_probas = NESW_merged.predict_proba([X[::4], X[1::4], X[2::4], X[3::4]], batch_size = 32)
