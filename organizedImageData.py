from matplotlib import cm
from scipy.misc import imresize, imsave, imread
import itertools
import os
import random
import cv2
import pandas as pd
import numpy as np
from imageProcessor import ColorDescriptor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from imageScraper import save_image, get_date
from coloradoGIS import load_geologic_history, load_features_and_shape
from coloradoGIS import find_which_feature
from imageAnalysisFunctions import corner_frac, surf, sklearn_hog
from IPython.display import HTML


def read_data(pickled_df_name):
    '''
    INPUT:  string: name of pickled DataFrame to be read
    OUTPUT: Pandas DataFrame from specified pickle object
    '''
    df = pd.read_pickle(pickled_df_name)
    return df


def merge_old_and_new(df_with_new, df):
    '''
    INPUT:  (1) Pandas DataFrame with new coordinates and labels
            (2) Old Pandas DataFrame
    OUTPUT: (1) Merged Pandas DataFrame
    '''
    co = df.columns.tolist()
    merged_df = pd.merge(left=df_with_new, right=df, how='left',
                         left_on=co[:4], right_on=co[:4])
    return merged_df


def download_images(df, loc='newdata'):
    '''
    INPUT:  (1) Pandas DataFrame with locations to download images
            (2) string: what folder to save new images to
    OUTPUT: None

    Download images from Google Maps Streetview API using
    the function defined in imageScraper.
    The API maxes out at 25000 images a day (6250 total locations,
    NESW for each location)
    '''
    how_many_images_downloaded = 0
    for lt, lg in zip(df['lat'][14244:], df['lng'][14244:]):
        for heading in [0, 90, 180, 270]:
            print how_many_images_downloaded
            how_many_images_downloaded += 1
            save_image((lt, lg), heading, loc=loc)


def write_mountains_cities_plains(df):
    '''
    INPUT:  (1) Pandas DataFrame
    OUTPUT: None

    This function will look through the locations and write a new column
    called 'mcp' (mountains cities plains) that somewhat arbitrarily makes
    a split between locations in Colorado that fall into one
    of these categories.

    MCP categorizations not used in final model, but were helpful in the
    beginning stages of training the model.
    '''
    cities = reduce(np.intersect1d, [np.where(df['elev'] > 1300),
                                     np.where(df['elev'] < 2400),
                                     np.where(df['lat'] > 38.6),
                                     np.where(df['lng'] > -105.25),
                                     np.where(df['lng'] < -104.2)])
    not_cities = np.setdiff1d(np.arange(len(df)), cities)
    plains = reduce(np.intersect1d, [not_cities,
                                     np.where(df['lng'] > -105.25),
                                     np.where(df['elev'] < 1800)])
    not_plains = np.setdiff1d(np.arange(len(df)), plains)
    mountains = reduce(np.intersect1d, [not_cities,
                                        not_plains,
                                        np.where(df['lng'] < -102)])
    category = ['mtn' if i in mountains
                else 'city' if i in cities
                else 'plains' for i in range(len(df))]
    df['mcp'] = category


def plot_3d(df, style='scatter', show_plot=True,
            options='normal', animate=False, to_html5=False):
    '''
    INPUT:  (1) Pandas DataFrame with locations
            (2) string: denoting the style. 'scatter' or 'wireframe'
                'scatter' looks significantly better.
            (3) boolean: show the plot?
            (4) string: 'mcp' or 'normal' corresponding to
                how to plot the points
                'mcp' will split into mountain, city, plains,
                'normal' will be all black
            (5) boolean: animate the plot?
            (6) boolean: return the animation as html5 tag?

    Plot all the locations in lat/lng/elev space.
    Just for fun to see all of the downloaded locations.
    '''
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)
    if show_plot:
        ax.set_xlabel('lat')
        ax.set_ylabel('lng')
        ax.set_zlabel('elevation (m)', rotation=90)
        cities = reduce(np.intersect1d, [np.where(df['elev'] > 1300),
                                         np.where(df['elev'] < 2400),
                                         np.where(df['lat'] > 38.6),
                                         np.where(df['lng'] > -105.25),
                                         np.where(df['lng'] < -104.2)])
        not_cities = np.setdiff1d(np.arange(len(df)), cities)
        plains = reduce(np.intersect1d,
                        [not_cities, np.where(df['lng'] > -105.25),
                         np.where(df['elev'] < 1800)])
        not_plains = np.setdiff1d(np.arange(len(df)), plains)
        mountains = reduce(np.intersect1d,
                           [not_cities, not_plains,
                            np.where(df['lng'] < -104.2)])
    if animate:
        def show():
            if style == 'scatter':
                if options == 'mcp':
                    ax.scatter(df.lat[cities],
                               df.lng[cities],
                               df.elev[cities],
                               color='k', s=1)
                    ax.scatter(df.lat[plains],
                               df.lng[plains],
                               df.elev[plains],
                               color='r', s=1)
                    ax.scatter(df.lat[mountains],
                               df.lng[mountains],
                               df.elev[mountains],
                               color='g', s=1)
                else:
                    ax.scatter(df.lat, df.lng, df.elev, color='k', s=1)

                ax.set_xlabel('Latitude ($^{\circ}$)')
                ax.set_ylabel('Longitude ($^{\circ}$)')
                ax.set_zlabel('Elevation (meters)')
                plt.gca().invert_yaxis()
            if style == 'wireframe':
                ordered_lat = df.lat[np.lexsort((df.lat.values,
                                                 df.lng.values))].values
                ordered_lng = df.lng[np.lexsort((df.lat.values,
                                                 df.lng.values))].values
                xv, yv = np.meshgrid(ordered_lat, ordered_lng)
                ordered_elev = df.elev[np.lexsort((df.lat.values,
                                                   df.lng.values))].values
                xe, ye = np.meshgrid(ordered_elev, ordered_elev)
                ax.plot_trisurf(df.lat.values[::10],
                                df.lng.values[::10],
                                df.elev.values[::10],
                                cmap=cm.jet, linewidth=0.2)
            plt.show()

        def animate(i):
            ax.view_init(elev=45., azim=i)

        anim = animation.FuncAnimation(fig, animate, init_func=show,
                                       frames=1080, interval=20, blit=False)
        anim.save('data_animation_rotation_dpi200_45deg_newlabels5.mp4',
                  fps=30, extra_args=['-vcodec', 'libx264'], dpi=200)
        if to_html5:
            html5_anim = animation.Animation.to_html5_video(anim)
            return html5_anim


def show_me(df, idx):
    '''
    INPUT:  (1) Pandas Dataframe with filenames written
            (2) integer: index of df to show
    OUTPUT: None

    Simple function to show the images at a given location.
    '''
    NESW = ['N', 'E', 'S', 'W']
    filenames = [df['base_filename'][idx] + cardinal_dir + '.png'
                 for cardinal_dir in NESW]
    print filenames
    fig = plt.figure(figsize=(16, 8))
    subplot_locs = [1, 2, 3, 4]
    titles = ['North', 'East', 'South', 'West']

    def add_ax(subplot_loc, title):
        ax = fig.add_subplot(2, 2, subplot_loc)
        ax.imshow(imread(filenames[subplot_loc - 1]))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    for subplot_loc, title in zip(subplot_locs, titles):
        add_ax(subplot_loc, title)
    print "The county is: {}".format(df['county'][idx])
    print "The true elevation is {}".format(df['elev'][idx])
    print "The age of the rock is in the range of {} million years".format(
            df['rock_age'][idx])
    plt.show()


def show_me_pano(df, idx, save=False):
    '''
    INPUT:  (1) Pandas Dataframe with filenames written
            (2) integer: index of df to show
            (3) boolean: save the image?
    OUTPUT: None

    Simple function to show the images at a given location in panorama form.
    Save the image if specified.
    '''
    NESW = ['N', 'E', 'S', 'W']
    filenames = [df['base_filename'][idx] + cardinal_dir + '.png'
                 for cardinal_dir in NESW]
    fig = plt.figure(figsize=(16, 3))

    def add_ax(filename, xloc, label):
        ax = fig.add_axes([xloc, .05, 1*1.31, .625*1.31])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(imread(filename))
        ax.set_title(label)
    labels = ['North', 'East', 'South', 'West']
    xlocs = np.linspace(-.530, .22, 4)
    for filename, xloc, label in zip(filenames, xlocs, labels):
        add_ax(filename, xloc, label)
    if save:
        plt.savefig('images_for_project_overview/pano_' + str(idx) +
                    '.png', dpi=150)
    plt.show()


def random_file_pull(df, yield_all_info=False):
    '''
    INPUT:  (1) Pandas DF
            (2) boolean: yield idx, direction and full filename?
                if False, yield just filename.

    Generate a random filename from the database. Good
    for testing different image processing functions. '''
    idx = np.random.randint(0, len(df))
    direction = random.choice(['N', 'E', 'S', 'W'])
    base = df.base_filename[idx]
    if yield_all_info:
        yield idx, direction, base + direction + '.png'
    else:
        yield base + direction + '.png'


def play_with_surf_and_cornerfrac(df):
    '''
    INPUT:  (1) Pandas DF
    OUTPUT: None

    Pull 20 images then calculate and show the SURF and corner
    fraction as computed using cv2. These functions are defined in
    imageAnalysisFunctions.
    '''
    some_random_files = [random_file_pull(df).next() for _ in range(20)]
    for img_name in some_random_files:
        surf(imread(img_name), show=True)
        corner_frac(imread(img_name), show=True)


def play_hog(df):
    '''
    INPUT:  (1) Pandas DF
    OUTPUT: None

    Pull 20 images, compute HOG, and display it.
    '''
    some_random_files = [random_file_pull(df, yield_all_info=True).next()
                         for _ in range(20)]
    for idx, direction, img_name in some_random_files:
        print idx, direction, img_name
        plt.clf()
        features, hog_img = sklearn_hog(img_name)
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(imread(img_name))
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(hog_img)
        fig.show()
    return hog_img


def play_color_likeness(df):
    '''
    INPUT:  (1) Pandas DF
    OUTPUT: None

    Pull 20 images and show the most similar images
    as found by euclidean distance from their color histograms.
    '''
    some_random_files = [random_file_pull(df, yield_all_info=True).next()
                         for _ in range(20)]
    for img_info_tuple in some_random_files:
        source_idx = img_info_tuple[0]
        direction = img_info_tuple[1]
        filename = img_info_tuple[2]
        column_direction = 'nearest_10_neighbors_euclidean_' + direction
        nearest_image_idx = df[column_direction][source_idx][0]
        nearest_image = (df['base_filename'][nearest_image_idx] +
                         direction + '.png')
        print "original image location is {}, {}".format(
                df['lat'][source_idx], df['lng'][source_idx])
        print "new image location is {}, {}".format(
                df['lat'][nearest_image_idx], df['lng'][nearest_image_idx])
        print "the indices in the df are {} and {}".format(
                source_idx, nearest_image_idx)
        print "\n"
        fig = plt.figure(figsize=(16, 8))

        # Show search image and most similar image in database #
        ax = fig.add_subplot(2, 2, 1)
        ax.imshow(imread(filename))
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(imread(nearest_image))
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Plot scatter of nearest 10 and a sampling of all datapoints #
        # See plot_nearest_10 for a better visualization #
        ax3 = fig.add_subplot(2, 2, 3)
        nearest_10 = find_locations_nearest_10(source_idx, direction)
        true_loc = df['lat'][source_idx], df['lng'][source_idx]
        ax3.scatter(nearest_10[2:, 1], nearest_10[2:, 0])
        ax3.scatter(true_loc[1], true_loc[0], color='#33FFFF',
                    label='true loc', s=30)
        ax3.scatter(nearest_10[0][1], nearest_10[0][0],
                    color='#00FF00', label='best guess', s=30)
        ax3.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2,
                   mode="expand", borderaxespad=0.)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.scatter(df['lng'][::10], df['lat'][::10])
        ax3.set_xlabel("Longitude")
        ax4.set_xlabel("Longitude")
        ax4.set_ylabel("Latitude")
        ax3.set_ylabel("Latitude")
        ax3.set_xlim(-109.5, -102.5)
        ax4.set_xlim(-109.5, -102.5)
        ax4.set_ylim(37, 41)
        ax3.set_ylim(37, 41)
        plt.show()


def find_locations_nearest_10(source_idx, direction):
    '''
    INPUT:  (1) integer: df index to compare to
            (2) string: 'N', 'E', 'S', or 'W'
    OUTPUT: (1) list of nearest locations
    '''
    column_direction = 'nearest_10_neighbors_euclidean_' + direction
    nearest_image_idx = df[column_direction][source_idx]
    nearest_locs = np.array([(df['lat'][idx], df['lng'][idx])
                            for idx in nearest_image_idx])
    return nearest_locs


def plot_nearest_10(df, idx, direction, save=False, loc='top_right'):
    '''
    INPUT:  (1) Pandas DataFrame with nearest_10 locs written
            (2) integer: index of the dataframe to plot
            (3) string: cardinal direction to plot ('N', 'E', 'S', or 'W')
            (4) boolean: save the plot?
            (5) loc: where to put the subplot
    OUTPUT: None

    This function will make a nicer version of the locations of the top 10
    closest locations (by color vector distance) than play_color_likeness.
    The plot will be saved if specified.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    true_loc = df['lat'][idx], df['lng'][idx]
    nearest_10 = find_locations_nearest_10(idx, direction)
    ax.scatter(true_loc[1], true_loc[0], color='#00FF00',
               label='True Location', s=120, zorder=3,
               marker='*', alpha=.7)
    ax.scatter(nearest_10[0][1], nearest_10[0][0], color='#088A29',
               label='Best Guess', s=50, zorder=2, alpha=.7)
    ax.scatter(nearest_10[1:, 1], nearest_10[1:, 0],
               label='Rest of Top 10', zorder=1, alpha=.7)
    ax.set_xlim(-109.5, -102.5)
    ax.set_ylim(37, 41)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if loc == 'top_right':
        ax2 = fig.add_axes([0.14, 0.55, .35, .34])
    elif loc == 'bottom_left':
        ax2 = fig.add_axes([0.54, 0.11, .35, .34])
    else:
        print 'Invalid subplot location!'

    def check_in_region(true_loc, test_loc_array, subplot_width):
        '''
        INPUT:  (1) float: true lat or lng (in degrees)
                (2) 1D numpy array: values to check if they are within
                    a subplot_width of (1)
                (3) float: width of subplot region (in degrees)
        OUTPUT: (1) 1D numpy array: True/False array corresponding
                    to whether each element in test_loc_array is within
                    a subplot width of the true location
        '''
        return ((test_loc_array < true_loc + subplot_width) &
                (test_loc_array > true_loc - subplot_width))
    subplot_width = 0.1  # in degrees
    idx_for_inset = np.where(check_in_region(true_loc[0],
                                             nearest_10[:, 0],
                                             subplot_width) &
                             check_in_region(true_loc[1],
                                             nearest_10[:, 1],
                                             subplot_width))[0]
    ax2.scatter(true_loc[1], true_loc[0], color='#00FF00',
                s=120, zorder=3, marker='*', alpha=.7)
    ax2.scatter(nearest_10[0][1], nearest_10[0][0], color='#088A29',
                s=50, zorder=2, alpha=.7)
    ax2.scatter(nearest_10[:, 1][idx_for_inset][1:],
                nearest_10[:, 0][idx_for_inset][1:], alpha=0.7)
    if loc == 'top_right':
        ax2.yaxis.tick_right()
    ax2.xaxis.tick_top()
    ax2.set_xlim(round(true_loc[1], 1) - subplot_width,
                 round(true_loc[1]) + subplot_width)
    ax2.set_ylim(round(true_loc[0], 1) - subplot_width,
                 round(true_loc[0]) + subplot_width)
    x_ticks = (np.linspace(round(true_loc[1], 1) - subplot_width,
                           round(true_loc[1], 1) + subplot_width, 5))
    y_ticks = (np.linspace(round(true_loc[0], 1) - subplot_width,
                           round(true_loc[0], 1) + subplot_width, 5))
    ax2.set_xticks(x_ticks)
    ax2.set_yticks(y_ticks)
    ax2.set_xticklabels([''] + [str(x) for x in x_ticks[1:-1]] + [''])
    ax2.set_xlim(round(true_loc[1], 2) - subplot_width,
                 true_loc[1] + subplot_width)
    ax2.set_ylim(round(true_loc[0], 2) - subplot_width,
                 true_loc[0] + subplot_width)
    ax.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=3,
              mode="expand", borderaxespad=0.)
    if save:
        plt.savefig('pano_likeness_' + str(idx) + '.png', dpi=200)
    plt.show()


def write_dates(df):
    '''
    INPUT:  (1) Pandas DF
    OUTPUT: None

    This function fills in missing dates to previously downloaded
    locations and images. The dates don't end up getting used anywhere
    in this iteration of modeling.
    '''
    df['date'] = [get_date(co) for co in itertools.izip(df.lat, df.lng)]


def write_filenames(df, options='data'):
    '''
    INPUT:  (1) Pandas DF
    OUTPUT: None

    Used to write the base filenames for each location.
    '''
    filename_lat = df.lat.apply(lambda x: str(x)[:8])
    filename_lng = df.lng.apply(lambda x: str(x)[:8])
    df['base_filename'] = (options + '/lat_' + filename_lat +
                           ',long_' + filename_lng + '_')


def write_shapefile_feature(df, options, index_to_fill=None):
    '''
    INPUT:  (1) Pandas DF
            (2) string: 'counties' or 'geologic_history'
                Will set pathway of which feature to write
            (3) 1D numpy array: the indices to fill, generally found
                using a np.where on where the classes haven't been written
    OUTPUT: None

    Analyze each long/lat point and write a feature based on its location.
    (either counties or rock age ranges). These features are
    derived from shapefiles downloaded from:
    http://coloradoview.org/cwis438/websites/ColoradoView/Data.php

    Pass the optional 'index to fill' to fill only certain rows, for example
    if the value was null, or if new data just came in.
    '''
    if options == 'counties':
        county_names_and_shapes = load_features_and_shape(options='counties')
        column_name = 'county'
        check_for_column(column_name)
        if index_to_fill is None:
            for idx, coord in enumerate(itertools.izip(df.lng.values,
                                                       df.lat.values)):
                print idx
                coun = find_which_feature(coord, county_names_and_shapes)
                print coun
                df[column_name][idx] = coun
        else:
            for idx in index_to_fill:
                coord = (df.ix[idx]['lng'], df.ix[idx]['lat'])
                print idx
                coun = find_which_feature(coord, county_names_and_shapes)
                print coun
                df[column_name][idx] = coun
    if options == 'geologic_history':
        rock_ages_and_shapes = load_features_and_shape(options=
                                                       'geologic_history')
        geologic_time_dictionary = load_geologic_history()
        column_name = 'rock_age'
        check_for_column(column_name)
        ranges = [0, 5, 20, 250, 3000]
        all_ranges = ['0-5', '5-20', '20-250', '250-3000']

        def which_range(ranges, all_ranges, rock_age):
            '''
            INPUT:  (1) list of numbers defining ranges
                    (2) list of ranges as strings
                    (3) integer: age of rock

            Find which range the given age is in.
            '''
            for age_idx, (r, r_plus1) in enumerate(zip(ranges[:-1],
                                                       ranges[1:])):
                try:
                    if ((rock_age > r) & (rock_age <= r_plus1)):
                        agerange_idx = age_idx
                except:
                    agerange_idx = 2  # 20-250 was a good middleground for nans
            try:
                return all_ranges[agerange_idx]
            except:
                return all_ranges[2]
        if index_to_fill is None:
            for idx, coord in enumerate(itertools.izip(df.lng.values,
                                                       df.lat.values)):
                rock_age_name = find_which_feature(coord, rock_ages_and_shapes)
                rock_age = geologic_time_dictionary[rock_age_name]
                rock_age_range = which_range(ranges, all_ranges, rock_age)
                print 'idx: {}, rock age: {}, range: {}'.format(
                        idx, rock_age, rock_age_range)
                df[column_name][idx] = rock_age_range
        else:
            for idx in index_to_fill:
                coord = (df.ix[idx]['lng'], df.ix[idx]['lat'])
                rock_age_name = find_which_feature(coord, rock_ages_and_shapes)
                rock_age = geologic_time_dictionary[rock_age_name]
                rock_age_range = which_range(ranges, all_ranges, rock_age)
                df[column_name][idx] = rock_age_range
                print 'idx: {}, rock age: {}, range: {}'.format(
                        idx, rock_age, rock_age_range)


def resize_and_save(df, img_name, true_idx, size='80x50', fraction=0.125):
    '''
    INPUT:  (1) Pandas DF
            (2) string: image name
            (3) integer: the true index in the df of the image
            (4) string: to append to filename
            (5) float: fraction to scale images by
    OUTPUT: None

    Resize and save the images in a new directory.
    Try to read the image. If it fails, download it to the raw data directory.
    Finally, read in the full size image and resize it.
    '''
    try:
        img = imread(img_name)
    except:
        cardinal_dir = img_name[-5:-4]
        cardinal_translation = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
        coord = (df.ix[true_idx]['lat'], df.ix[true_idx]['lng'])
        print 'Saving new image...'
        print coord, cardinal_dir, cardinal_translation[cardinal_dir]
        save_image(coord, cardinal_translation[cardinal_dir], loc='newdata')
    finally:
        img_name_to_write = ('newdata_' + size + '/' +
                             img_name[8:-4] + size + '.png')
        if os.path.isfile(img_name_to_write) == False:
            img = imread(img_name)
            resized = imresize(img, fraction)
            print 'Writing file...'
            imsave(img_name_to_write, resized)


def resize_all_images(df, start_idx=0):
    '''
    INPUT:  (1) Pandas DF
            (2) integer: index in df to start resizing images at. If the
                image has already been resized it will not resize it again;
                the file path is checked.
    OUTPUT: None
    '''
    NESW = ['N', 'E', 'S', 'W']
    for idx in range(df[start_idx:].shape[0]):
        true_idx = start_idx + idx
        print idx, true_idx
        for cardinal_dir in NESW:
            image_name = (df.ix[true_idx]['base_filename'] +
                          cardinal_dir + '.png')
            resize_and_save(df, image_name, true_idx)


def check_for_column(column_name, typ=object):
    '''
    INPUT:  (1) string: the name of the column to check for
            (2) type: the data type to format the
                column as (object, float, etc.)
    OUTPUT: None

    Check the dataframe for the presence of the column. Create
    it and make it of the specified type if it doesn't exist. '''
    if column_name not in df.columns:
        df[column_name] = 0.
        df[column_name] = df[column_name].astype(typ)


def calculate_features_and_determine_closest(df, cd,
                                             distance_metric='euclidean'):
    '''
    INPUT:  (1) Pandas DF
            (2) ColorDescriptor object
            (3) string: the distance metric to use:
                either 'euclidean' or 'cosine'

    Calculate the ColorDescriptor histogram to a column in the dataframe.
    Store in a temporary numpy array, calculate the euclidean distance for each
    pair, and write the indices of the closest 10 images in colorspace by
    cardinal direction.
    '''
    NESW = ['N', 'E', 'S', 'W']
    all_images_count = df.shape[0]
    for cardinal_dir in NESW[-1]:
        ltlg_features = None
        column_name = ('nearest_10_neighbors_' +
                       distance_metric + '_' + cardinal_dir)
        for idx in range(all_images_count):
            print 'Writing features for index {}'.format(idx)
            image_name = df.iloc[idx]['base_filename'] + cardinal_dir + '.png'
            image = imread(image_name)
            if ltlg_features is None:
                ltlg_features = np.array(cd.describe(image))
            else:
                ltlg_features = np.vstack((ltlg_features, cd.describe(image)))
        if distance_metric == 'euclidean':
            check_for_column(column_name)
            for idx in range(all_images_count):
                print idx
                distance = np.linalg.norm(ltlg_features[idx] -
                                          ltlg_features, axis=1)
                df[column_name][idx] = np.argsort(distance)[1:11]
        if distance_metric == 'cosine':
            check_for_column(column_name)
            for idx in range(all_images_count):
                print idx
                numerator = np.dot(ltlg_features[idx], ltlg_features.T)
                denominator = (np.linalg.norm(ltlg_features[0]) *
                               np.linalg.norm(ltlg_features, axis=1))
                cosine_sim = numerator/denominator
                df[column_name][idx] = np.argsort(cosine_sim)[-11:-1][::-1]


def calc_avg_color(df):
    '''
    INPUT:  (1) Pandas DF
    OUTPUT: None

    Calculate the average color for a given location over the N, E, S, and W
    images for both the top and bottom halves of the images. Save the values
    to columns in the dataframe to be plotted (by county) later on.
    '''
    NESW = ['N', 'E', 'S', 'W']
    cols_to_write = ['avg_r_up', 'avg_r_low', 'avg_g_up',
                     'avg_g_low', 'avg_b_up', 'avg_b_low']
    for col in cols_to_write:
        check_for_column(col)
    for idx in range(df.shape[0]):
        print idx
        r_up, r_low, g_up, g_low, b_up, b_low = [], [], [], [], [], []
        for cardinal_dir in NESW:
            im = imread(df['base_filename'][idx] + cardinal_dir + '.png')
            r_up += [im[:200, :, 0].mean()]
            r_low += [im[200:, :, 0].mean()]
            g_up += [im[:200, :, 1].mean()]
            g_low += [im[200:, :, 1].mean()]
            b_up += [im[:200, :, 2].mean()]
            b_low += [im[200:, :, 2].mean()]
        df['avg_r_up'][idx] = np.mean(r_up)
        df['avg_r_low'][idx] = np.mean(r_low)
        df['avg_g_up'][idx] = np.mean(g_up)
        df['avg_g_low'][idx] = np.mean(g_low)
        df['avg_b_up'][idx] = np.mean(b_up)
        df['avg_b_low'][idx] = np.mean(b_low)


if __name__ == '__main__':
    df = read_data()
