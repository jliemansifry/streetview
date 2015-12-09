from matplotlib import cm
import itertools
import random
# import itertools
# import cv2
import pandas as pd
import numpy as np
# from imageProcessor import ColorDescriptor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from imageScraper import save_image, get_date
from coloradoGIS import load_geologic_history, load_features_and_shape, find_which_feature
from imageAnalysisFunctions import corner_frac, surf, cv2_image, sklearn_hog
# cd = ColorDescriptor((8, 12, 3))

def read_data():
    # df = pd.read_csv('big_list_with_filenames.csv')
    # df = pd.read_csv('big_list_with_road_colorhists.csv')
    # df = pd.read_pickle('big_list_with_nearest_10.pkl')
    df = pd.read_csv('big_list_o_trimmed_coord.csv')
    return df

def download_images(df):
    ''' Download images from Google Maps Streetview API using
    the function defined in imageScraper. Must be used in bunches, 
    as the API maxes out at 2500 images a day (625 total locations,
    NESW for each location)'''
    count = 0
    for lt, lg in zip(df['lat'][12275:12900], df['lng'][12275:12900]):
        for heading in [0, 90, 180, 270]:
            print count
            count += 1
            save_image((lt,lg), heading)

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

def random_file_pull(df, yield_all_info = False):
    ''' Generate a random filename from the database. Good
    for testing different image processing functions. '''
    idx = np.random.randint(0,len(df))
    direction = random.choice(['N','E','S','W'])
    base = df.base_filename[idx]
    if yield_all_info:
        yield idx, direction, base + direction + '.png'
    else:
        yield base + direction + '.png'

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
        # plt.clf()
        fig = plt.figure(figsize = (16,8))
        ax = fig.add_subplot(1,2,1)
        ax.imshow(cv2_image(img_name))
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(hog_img)
        plt.show()
    return hog_img

def play_color_likeness(df):
    ''' Pull 20 images and show the most similar images 
    as found by euclidean distance from their color histograms'''
    some_random_files = [random_file_pull(df, yield_all_info = True).next() for _ in range(20)]
    for img_info_tuple in some_random_files:
        source_idx = img_info_tuple[0]
        direction = img_info_tuple[1]
        filename = img_info_tuple[2]
        column_direction = 'nearest_10_neighbors_euclidean_' + direction
        nearest_image_idx = df[column_direction][source_idx][0]
        nearest_image = df['base_filename'][nearest_image_idx] + direction + '.png'
        print "original image location is {}, {}".format(df['lat'][source_idx], df['lng'][source_idx])
        print "new image location is {}, {}".format(df['lat'][nearest_image_idx], df['lng'][nearest_image_idx])
        print "the indices in the df are {} and {}".format(source_idx, nearest_image_idx)
        print "\n"
        fig = plt.figure(figsize = (16,8))
        ax = fig.add_subplot(2,2,1)
        ax.imshow(cv2_image(filename))
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(cv2_image(nearest_image))
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3 = fig.add_subplot(2,2,3)
        nearest_10 = find_locations_nearest_10(source_idx, direction)
        true_loc = df['lat'][source_idx], df['lng'][source_idx]
        ax3.scatter(nearest_10[2:,1], nearest_10[2:,0])#, label = 'rest of top 10')
        ax3.scatter(true_loc[1], true_loc[0], color = '#33FFFF', label = 'true loc', s = 30)
        ax3.scatter(nearest_10[0][1], nearest_10[0][0], color = '#00FF00', label = 'best guess', s = 30)
        # ax3.legend(bbox_to_anchor=(-.5, 1.0))
        ax3.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        ax4 = fig.add_subplot(2,2,4)
        ax4.scatter(df['lng'][::10], df['lat'][::10])
        ax3.set_xlabel("Longitude")
        ax3.set_ylabel("Latitude")
        ax4.set_xlabel("Longitude")
        ax4.set_ylabel("Latitude")
        ax3.set_xlim(-109.5, -102.5)
        ax3.set_ylim(37, 41)
        ax4.set_xlim(-109.5, -102.5)
        ax4.set_ylim(37, 41)
        plt.show()

def find_locations_nearest_10(source_idx, direction):
    column_direction = 'nearest_10_neighbors_euclidean_' + direction
    nearest_image_idx = df[column_direction][source_idx]
    nearest_locs = np.array([(df['lat'][idx], df['lng'][idx]) for idx in nearest_image_idx])
    return nearest_locs

def write_dates(df):
    ''' Filled in missing dates after downloading images before 
    get_date was working. Might want to use something like this
    later to write vectors for each image. '''
    df['date'] = [get_date(co) for co in itertools.izip(df.lat, df.lng)]

def write_filenames(df):
    ''' Used to write the base filenames for each location.'''
    filename_lat = df.lat.apply(lambda x: str(x)[:8])
    filename_lng = df.lng.apply(lambda x: str(x)[:8])
    df['base_filename'] = 'data/lat_' + filename_lat + ',long_' + filename_lng + '_'

def write_shapefile_feature(df, options):
    ''' Analyze each long/lat point and write a feature based on its location.
    (either counties or rock age ranges). These features are
    derived from shapefiles downloaded from:
    http://coloradoview.org/cwis438/websites/ColoradoView/Data.php '''
    if options == 'counties':
        county_names_and_shapes = load_features_and_shape(options = 'counties')
        column_name = 'county'
        check_for_column(column_name)
        for idx, coord in enumerate(itertools.izip(df.lng.values, df.lat.values)):
            print idx
            coun = find_which_feature(coord, county_names_and_shapes)
            print coun
            df[column_name][idx] = coun
    if options == 'geologic_history':
        rock_ages_and_shapes = load_features_and_shape(options = 'geologic_history')
        geologic_time_dictionary = load_geologic_history()
        column_name = 'rock_age'
        check_for_column(column_name)
        ranges = [0, 5, 20, 250, 500, 3000]
        all_ranges = ['0-5', '5-20', '20-250', '250-500', '500-3000']
        for idx, coord in enumerate(itertools.izip(df.lng.values, df.lat.values)):
            print idx
            rock_age_name = find_which_feature(coord, rock_ages_and_shapes)
            rock_age = geologic_time_dictionary[rock_age_name]
            print rock_age
            for age_idx, (r, r_plus1) in enumerate(zip(ranges[:-1], ranges[1:])):
                try:
                    if ((rock_age > r) & (rock_age <= r_plus1)):
                        agerange_idx = age_idx
                except: 
                    agerange_idx = 2 #20-250 was a good middleground for nans
            rock_age_range = all_ranges[agerange_idx]
            print rock_age_range
            df[column_name][idx] = rock_age_range

def check_for_column(column_name, typ = object):
    ''' Check the dataframe for the presence of the column. Create
    it and make it of the specified type. '''
    if column_name not in df.columns:
        df[column_name] = 0.
        df[column_name] = df[column_name].astype(typ)

def write_features(df, cd):
    ''' Write the ColorDescriptor histogram to a column in the dataframe.
    Not worth actually using, as unpacking the feature vectors from each
    column is a pain, and storing them isn't really necessary. '''
    NESW = ['N', 'E', 'S', 'W']
    for cardinal_dir in NESW:
        column_name = 'hist_vec_' + cd.hist_loc + '_' + cardinal_dir
        check_for_column(column_name)
    for idx in range(df.shape[0]):
        print idx
        for cardinal_dir in NESW:
            image_name = df.iloc[idx]['base_filename'] + cardinal_dir + '.png'
            image = cv2_image(image_name)
            ltlg_features = np.array(cd.describe(image))
            df['hist_vec_' + cd.hist_loc + '_' + cardinal_dir][idx] = ltlg_features

def calculate_features_and_determine_closest(df, cd, distance_metric = 'euclidean'):
    ''' Calculate the ColorDescriptor histogram to a column in the dataframe. 
    Store in a temporary numpy array, calculate the euclidean distance for each
    pair, and write the indices of the closest 10 images in colorspace by 
    cardinal direction. '''
    NESW = ['N', 'E', 'S', 'W']
    all_images_count = df.shape[0]
    for cardinal_dir in NESW[-1]:
        ltlg_features = None
        for idx in range(all_images_count):
            print idx
            image_name = df.iloc[idx]['base_filename'] + cardinal_dir + '.png'
            image = cv2_image(image_name)
            if ltlg_features is None:
                ltlg_features = np.array(cd.describe(image))
            else:
                ltlg_features = np.vstack((ltlg_features, cd.describe(image)))
        if distance_metric == 'euclidean':
            column_name = 'nearest_10_neighbors_' + distance_metric + '_' + cardinal_dir
            check_for_column(column_name)
            for idx in range(all_images_count):
                print idx
                distance = np.linalg.norm(ltlg_features[idx] - ltlg_features, axis = 1)
                df['nearest_10_neighbors_' + distance_metric + '_' + cardinal_dir][idx] = np.argsort(distance)[1:11]
        if distance_metric == 'cosine':
            # turns out cosine distance returns the same 'nearest' images as euclidean distance
            column_name = 'nearest_10_neighbors_' + distance_metric + '_' + cardinal_dir
            check_for_column(column_name)
            for idx in range(all_images_count):
                print idx
                numerator = np.dot(ltlg_features[idx], ltlg_features.T)
                denominator = np.linalg.norm(ltlg_features[0]) * np.linalg.norm(ltlg_features, axis = 1)
                df['nearest_10_neighbors_' + distance_metric + '_' + cardinal_dir][idx] = np.argsort(numerator/denominator)[-11:-1][::-1]

if __name__ == '__main__':
    # pass
    df = read_data()
    # test = plot_3d(df, style = 'wireframe', show = False)
    # write_filenames(df)
    # df.to_csv('big_list_with_filenames.csv', index = False)
    # df.to_csv('big_list_with_filenames.csv', index = False)
