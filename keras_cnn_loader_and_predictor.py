import theano
from organizedImageData import write_filenames
import pandas as pd
import numpy as np
from scipy.misc import imread
from keras.models import model_from_json
from coloradoGIS import plot_shapefile


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
    null_categories = np.where(pd.isnull(categories))[0][0]
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
                image_class = category_list.index(cnty)
            image_data = imread(image_name)
            idx_to_write = df_idx * unique_directions + sub_idx
            all_X_data[idx_to_write] = image_data
            all_y_data[idx_to_write] = image_class
            sub_idx += 1
    
    all_X_data = all_X_data.reshape(all_X_data.shape[0], 3, 50, 80)
    all_X_data.astype('float32')
    all_X_data /= 255.
    return df, all_X_data, all_y_data, category_name, categories, category_list, unique_count


def load_iPhone_images():
    X = np.zeros((4, 50, 80, 3))
    img_nums = range(5908, 5910) + range(5911, 5921)
    img_names = ['/Users/jliemansifry/Desktop/outside_galvanize_test/IMG_' +
                 str(num) + '.jpg'
                 for num in img_nums]
    for idx, img_name in enumerate(img_names):
        X[idx] = imread(img_name)
    X = X.reshape(X.shape[0], 3, 50, 80)
    X = X.astype('float32')
    X /= 255.
    return X


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
    input_layer = model.layers[0].input
    specified_layer_output = model.layers[layer].get_output(train=False)
    get_activations = theano.function([input_layer],
                                      specified_layer_output,
                                      allow_input_downcast=True)
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
    final_layer = np.vstack((N_activations, E_activations,
                             S_activations, W_activations))
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


def return_specified_proba(X, idx, model_name, categories, df,
                           NESW_merged=None, local=False,
                           show=False, save=False, two_sets_of_NESW=False):
    '''
    INPUT:  (1) 4D numpy array: All X data
            (2) integer: index to determine probabilities for
            (3) string: the full path to the model name
            (4) list: all categories
            (5) df: to determine the true county
            (6) optional previously loaded model
            (7) boolean: using local images or not?
            (8) boolean: show the probabilities on a map?
            (9) boolean: save the image?
            (10) boolean: two sets of NESW models as input?
    OUTPUT: (1) dict: categories and their associated probabilities

    Return the probabilities associated with each category that the model
    has been trained on for a specific location in the dataset.
    If no model has been specified, return the model as well so that it does
    not need to be reloaded in the future.
    '''
    model_built = False
    f = 'shapefiles/Shape/GU_CountyOrEquivalent'
    if NESW_merged is None:
        model_built = True
        NESW_merged = build_merged_model(model_name)
    N_idx = idx * 4
    E_idx = idx * 4 + 1
    S_idx = idx * 4 + 2
    W_idx = idx * 4 + 3
    end_idx = idx * 4 + 4
    if two_sets_of_NESW:
        final_probas = NESW_merged.predict_proba([X[N_idx:end_idx:4],
                                                  X[E_idx:end_idx:4],
                                                  X[S_idx:end_idx:4],
                                                  X[W_idx:end_idx:4],
                                                  X[N_idx:end_idx:4],
                                                  X[E_idx:end_idx:4],
                                                  X[S_idx:end_idx:4],
                                                  X[W_idx:end_idx:4]],
                                                 batch_size=1)[0]
    else:
        final_probas = NESW_merged.predict_proba([X[N_idx:end_idx:4],
                                                  X[E_idx:end_idx:4],
                                                  X[S_idx:end_idx:4],
                                                  X[W_idx:end_idx:4]],
                                                 batch_size=1)[0]
    probas_dict = {c: p for c, p in zip(categories, final_probas)}
    if show:
        print 'showing'
        plot_shapefile(f, options='counties', more_options='by_probability',
                       cm='continuous', df=df, probas_dict=probas_dict,
                       local=local, true_idx=idx, show=True, save=False)
    if save:
        print 'saving'
        plot_shapefile(f, options='counties', more_options='by_probability',
                       cm='continuous', df=df, probas_dict=probas_dict,
                       local=local, true_idx=idx, show=False, save=True)
    if model_built:
        return probas_dict, NESW_merged


def calc_top_n_acc(X, y, NESW_merged, n, idx, end_idx, two_sets_of_NESW=True):
    '''
    INPUT:  (1) 4D numpy array: all X data
            (2) 1D numpy array: all y data
            (3) Keras model object
            (4) integer to calculate top_n_class accuracy up to
            (5) integer: start idx
            (6) integer: end idx
            (7) boolean: two sets of NESW models as input?
    OUTPUT: (1) float: the top n probability
    '''
    N_idx = idx * 4
    E_idx = idx * 4 + 1
    S_idx = idx * 4 + 2
    W_idx = idx * 4 + 3
    end_idx = end_idx * 4
    if two_sets_of_NESW:
        probas = NESW_merged.predict_proba([X[N_idx:end_idx:4],
                                            X[E_idx:end_idx:4],
                                            X[S_idx:end_idx:4],
                                            X[W_idx:end_idx:4],
                                            X[N_idx:end_idx:4],
                                            X[E_idx:end_idx:4],
                                            X[S_idx:end_idx:4],
                                            X[W_idx:end_idx:4]],
                                           batch_size=32)
    else:
        probas = NESW_merged.predict_proba([X[N_idx:end_idx:4],
                                            X[E_idx:end_idx:4],
                                            X[S_idx:end_idx:4],
                                            X[W_idx:end_idx:4]],
                                           batch_size=32)
    y_true = y[N_idx:end_idx:4]
    top_n_classes = np.fliplr(np.argsort(probas, axis=1))[:, :n]
    in_top_n = [1 if y_true[row_idx] in row
                else 0
                for row_idx, row in enumerate(top_n_classes)]
    return np.sum(in_top_n) / float(len(y_true))


if __name__ == '__main__':
    df, X, y, category_name, categories65, categories64, unique_count = load_data()
