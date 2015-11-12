from matplotlib import cm
import random
import itertools
import cv2
import pandas as pd
import numpy as np
from imageProcessorRoadLike import ColorDescriptor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from imageScraper import save_image
from imagePlaytime import corner_frac, surf, cv2_image, cv2_hog, sklearn_hog
#df = pd.read_csv('big_list_o_updated_coords.csv', names = ['lat', 'lng', 'elev', 'city', 'county', 'state', 'full_address', 'date'])
#df = pd.read_csv('big_list_o_updated_coords.csv') #, names = ['lat', 'lng', 'elev', 'city', 'county', 'state', 'full_address', 'date'])
#df.drop(['city','county','state'], axis=1, inplace = True)
#df.to_csv('big_list_o_trimmed_coords.csv', index = False)

cd = ColorDescriptor((8, 12, 3))

def read_data():
    # df = pd.read_csv('big_list_o_trimmed_coords.csv')
    df = pd.read_csv('big_list_with_filenames.csv')
    return df

def download_images(df):
    count = 0
    for lt, lg in zip(df['lat'][7200:7512], df['lng'][7200:7513]): # currently done through image 3500
        # make a catchsafe for overwriting images!!!
        for heading in [0, 90, 180, 270]:
            print count
            count += 1
            save_image((lt,lg), heading)

def write_dates(df):
    ''' might want to use this later to write vectors for each image'''
    from imageScraper import get_date
    df['date'] = [get_date(co) for co in (zip([lat for lat in df['lat']],[lng for lng in df['lng']]))]

def plot_3d(df, style = 'scatter', show = True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # c = np.array(6*['b','g','r']).reshape(6,3)
    # s = np.array(3*num_hidden).reshape((3,6)).T
    ax.set_xlabel('lat') 
    ax.set_ylabel('lng')
    ax.set_zlabel('elevation (m)', rotation = 90)
    #plt.zlabel('loss')
    if show:
        if style == 'scatter':
            ax.scatter(df.lat, df.lng, df.elev, s = 1)
        if style == 'wireframe':
            ordered_lat = df.lat[np.lexsort((df.lat.values, df.lng.values))].values
            ordered_lng = df.lng[np.lexsort((df.lat.values, df.lng.values))].values
            xv, yv = np.meshgrid(ordered_lat, ordered_lng)
            ordered_elev = df.elev[np.lexsort((df.lat.values, df.lng.values))].values
            xe, ye = np.meshgrid(ordered_elev, ordered_elev)
            ax.plot_trisurf(df.lat.values[::10], df.lng.values[::10], df.elev.values[::10], cmap=cm.jet, linewidth=0.2) 
        plt.show()

def make_hist(img):
    return cd.describe(img)

def random_file_pull(df):
    base = df.base_filename[np.random.randint(0,len(df))]
    # direction_cycler = itertools.cycle(['N','E','S','W'])
    yield base + random.choice(['N','E','S','W']) + '.png'

def play_with_surf_and_cornerfrac(df):
    some_random_files = [random_file_pull(df).next() for _ in range(20)]
    for img_name in some_random_files:
        surf(cv2_image(img_name), show = True)
        corner_frac(cv2_image(img_name), show = True)

def play_hog(df):
    some_random_files = [random_file_pull(df).next() for _ in range(20)]
    for img_name in some_random_files:
        features, hog_img = sklearn_hog(img_name)
        print max(features)
        fig = plt.figure(figsize = (16,8))
        plt.clf()
        ax = fig.add_subplot(1,2,1)
        ax.imshow(cv2_image(img_name))
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(hog_img)
        plt.show()
    return hog_img


## look into cv2 basic image processing algorithms. edge highlighting, 
# that kind of thing. might be helpful for when misuing NN. 
# cross correlate images with features to highlight features? 

def write_filenames(df):
    filename_lat = df.lat.apply(lambda x: str(x)[:8])
    filename_lng = df.lng.apply(lambda x: str(x)[:8])
    # df['base_filename'] = 'data/lat_' + df.lat.map(str) + ',long_' + df.lng.map(str) + '_'
    df['base_filename'] = 'data/lat_' + filename_lat + ',long_' + filename_lng + '_'

if __name__ == '__main__':
    df = read_data()
    # test = plot_3d(df, style = 'wireframe', show = False)
    # write_filenames(df)
    # df.to_csv('big_list_with_filenames.csv', index = False)
