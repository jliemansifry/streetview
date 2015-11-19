from matplotlib import cm
import random
import itertools
import cv2
import pandas as pd
import numpy as np
from imageProcessor import ColorDescriptor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from imageScraper import save_image, get_date
from imageAnalysisFunctions import corner_frac, surf, cv2_image, sklearn_hog

# cd = ColorDescriptor((8, 12, 3))

def read_data():
    df = pd.read_csv('big_list_with_filenames.csv')
    return df

def download_images(df):
    ''' Download images from Google Maps Streetview API using
    the function defined in imageScraper. Must be used in bunches, 
    as the API maxes out at 2500 images a day (625 total locations,
    NESW for each location)'''
    count = 0
    for lt, lg in zip(df['lat'][7200:7512], df['lng'][7200:7513]):
        for heading in [0, 90, 180, 270]:
            print count
            count += 1
            save_image((lt,lg), heading)

def write_dates(df):
    ''' Filled in missing dates after downloading images before 
    get_date was working. Might want to use something like this
    later to write vectors for each image. '''
    df['date'] = [get_date(co) for co in (zip([lat for lat in df['lat']],[lng for lng in df['lng']]))]

def plot_3d(df, style = 'scatter', show = True):
    ''' Plot all the locations in lat/lng/elev space. 
    Just for fun to see all of the downloaded locations. '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('lat') 
    ax.set_ylabel('lng')
    ax.set_zlabel('elevation (m)', rotation = 90)
    if show:
        if style == 'scatter':
            ax.scatter(df.lat, df.lng, df.elev, s = 1)
            plt.gca().invert_yaxis()
        if style == 'wireframe':
            ordered_lat = df.lat[np.lexsort((df.lat.values, df.lng.values))].values
            ordered_lng = df.lng[np.lexsort((df.lat.values, df.lng.values))].values
            xv, yv = np.meshgrid(ordered_lat, ordered_lng)
            ordered_elev = df.elev[np.lexsort((df.lat.values, df.lng.values))].values
            xe, ye = np.meshgrid(ordered_elev, ordered_elev)
            ax.plot_trisurf(df.lat.values[::10], df.lng.values[::10], df.elev.values[::10], cmap=cm.jet, linewidth=0.2) 
        plt.show()

def make_hist(cd, img):
    ''' Make a 3D HSV histogram as described by the chosen
    ColorDescriptor object that is passed. 'Roadlike' and 'quadrant'
    are the options for the ColorDescriptor; their resulting
    histogram feature vectors will be different lengths. '''
    return cd.describe(img)

def random_file_pull(df):
    ''' Generate a random filename from the database. Good
    for testing different image processing functions. '''
    base = df.base_filename[np.random.randint(0,len(df))]
    yield base + random.choice(['N','E','S','W']) + '.png'

def play_with_surf_and_cornerfrac(df):
    ''' Pull 20 images then calculate and show the SURF and corner fraction
    as computed using cv2.'''
    some_random_files = [random_file_pull(df).next() for _ in range(20)]
    for img_name in some_random_files:
        surf(cv2_image(img_name), show = True)
        corner_frac(cv2_image(img_name), show = True)

def play_hog(df):
    ''' Pull 20 images, compute HOG, and display it. '''
    some_random_files = [random_file_pull(df).next() for _ in range(20)]
    for img_name in some_random_files:
        features, hog_img = sklearn_hog(img_name)
        plt.clf()
        fig = plt.figure(figsize = (16,8))
        ax = fig.add_subplot(1,2,1)
        ax.imshow(cv2_image(img_name))
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(hog_img)
        plt.show()
    return hog_img

def write_filenames(df):
    ''' Used to write the base filenames for each location.'''
    filename_lat = df.lat.apply(lambda x: str(x)[:8])
    filename_lng = df.lng.apply(lambda x: str(x)[:8])
    df['base_filename'] = 'data/lat_' + filename_lat + ',long_' + filename_lng + '_'

def write_features(df, cd):
    ''' Write the ColorDescriptor histogram to a column in the dataframe. '''
    NESW = ['N', 'E', 'S', 'W']
    for cardinal_dir in NESW:
        if 'hist_vec_' + cd.hist_loc + '_' + cardinal_dir not in df.columns:
            df['hist_vec_' + cd.hist_loc + '_' + cardinal_dir] = 0.
            df['hist_vec_' + cd.hist_loc + '_' + cardinal_dir] = df['hist_vec_' + cd.hist_loc + '_' + cardinal_dir].astype(object)
    for idx in xrange(0, 2):
        for cardinal_dir in NESW:
            image_name = df.iloc[idx]['base_filename'] + cardinal_dir + '.png'
            image = cv2_image(image_name)
            ltlg_features = np.array(cd.describe(image))
            df['hist_vec_' + cd.hist_loc + '_' + cardinal_dir][idx] = ltlg_features

if __name__ == '__main__':
    df = read_data()
    # test = plot_3d(df, style = 'wireframe', show = False)
    # write_filenames(df)
    # df.to_csv('big_list_with_filenames.csv', index = False)
