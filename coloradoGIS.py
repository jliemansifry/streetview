import pandas as pd
import cv2
from mpl_toolkits.basemap import Basemap
from pyproj import Proj, transform
import fiona
from fiona.crs import from_epsg
import numpy as np
import matplotlib.pyplot as plt
import shapely
import shapefile
import matplotlib
from matplotlib.patches import Polygon, PathPatch, Patch
from matplotlib.collections import PatchCollection, LineCollection
from scipy.misc import imread
from imagePresentationFunctions import make_cmyk_greyscale_continuous_cmap
import random
import shapely.geometry as sg

def plot_shapefile(f, options = 'counties', more_options = None, cm = 'blues', df = None, probas_dict = None, true_idx = None):
    ''' 
    INPUT:  (1) string: shapefile to use
            (2) string: options to specify that build a nice plot
                        'counties' or 'rocktypes' or 'geologic_history' 
                        'counties' will plot the counties of just CO, 
                        even though this shapefile includes some counties
                        in neighboring states
                        'rocktypes' will plot all distinct rocktypes in CO
                        'geologic_history' plots the rocktypes by age
                        rather than unique type (4 age ranges rather than
                        29 primary rocktypes)
            (3) string: more options that specify for counties what colors 
                        to plot
                        'by_img_color' or 'by_probability'
                        'by_img_color' will plot the avg color of each img
                        'by_probability' will plot scale the colormap to 
                        reflect the probability that a given image is 
            (4) string: colormap to specify
                        'blues' or 'continuous'
                        'blues' is easy on the eyes for random assignment
                        'continuous' is good for probabilities
                        '''
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    m = Basemap(width=800000,height=550000, resolution='l', projection='aea',
                lat_1=37.,lat_2=41,lon_0=-105.55,lat_0=39)
    m.readshapefile(f, name = 'state', color = 'none')
    
    ## OPTIONS ##
    if options == 'rocktypes':
        rocks= np.unique([shape_info['ROCKTYPE1']
                          for shape_info in m.state_info])
        num_colors = len(rocks)
    elif options == 'geologic_history':
        geologic_time_dictionary = load_geologic_history()
        ranges = [0, 5, 20, 250, 3000] # in Myr
        all_ranges = []
        for r, r_plus1 in zip(ranges[:-1], ranges[1:]):
            all_ranges += [str(r) + '-' + str(r_plus1)]
        num_colors = len(all_ranges)
    elif more_options == 'by_probability':
        max_proba = max(probas_dict.values())
        num_colors = 101
        proba_range = np.linspace(0.0, max_proba, num_colors)

    ## COLOR MAPS ##
    if cm == 'blues':
        cmap = plt.get_cmap('Blues')
    elif cm == 'continuous':
        cmap = make_cmyk_greyscale_continuous_cmap()
    else:
        cmap = plt.get_cmap(cm)
    discrete_colormap = [cmap(1.*i/num_colors) for i in range(num_colors)]

    ## LOOP THROUGH SHAPEFILES ##
    for info, shape in zip(m.state_info, m.state):
        pc = patch_collection(shape)

        ## OPTIONS OF HOW TO PLOT THE SHAPEFILE##
        if options == 'counties':
            county_name = info['COUNTY_NAM']
            state_name = info['STATE_NAME']
            if state_name != 'Colorado':
                continue # ignore shapefiles from out of CO

            ## MORE OPTIONS FOR COUNTIES ##
            if more_options == 'by_img_color':
                locs = np.where(df['county'] == county_name)[0]
                r_avg = df['avg_r_low'][locs].mean()
                g_avg = df['avg_g_low'][locs].mean()
                b_avg = df['avg_b_low'][locs].mean()
                pc.set_color((r_avg/255., g_avg/255., b_avg/255.))
            elif more_options == 'by_probability':
                proba = probas_dict[county_name]
                proba_idx = int(proba / max_proba * 100)
                pc.set_color(discrete_colormap[proba_idx])
                pc.set_edgecolor('k')
                if county_name == df['county'][true_idx]:
                    pc.set_hatch('//')
                    pc.set_edgecolor('w')
                    pc.set_hatch('\\')
                    pc.set_edgecolor('k')
            else:
                pc.set_color(random.choice(discrete_colormap))
        elif options == 'rocktypes':
            rocktype = info['ROCKTYPE1']
            idx = np.where(rocks == rocktype)[0][0]
            pc.set_color(discrete_colormap[idx])
        elif options == 'geologic_history':
            rock_age = info['UNIT_AGE']
            for index, (r, r_plus1) in enumerate(zip(ranges[:-1], ranges[1:])):
                try:
                    if ((geologic_time_dictionary[rock_age] > r) & 
                    (geologic_time_dictionary[rock_age] < r_plus1)):
                        idx = index 
                except: 
                    idx = 2 #20-250 was a good middleground for nans
            pc.set_color(discrete_colormap[idx])
        elif options == 'nofill':
            pc.set_facecolor('none')
        else:
            pc.set_color(random.choice(discrete_colormap))
        if more_options == 'by_probability':
            ax2 = fig.add_axes([0.78, 0.1, 0.03, 0.8])
            cb = matplotlib.colorbar.ColorbarBase(ax2, cmap = cmap, 
                    ticks = proba_range, boundaries = proba_range, format = '%1i')
            labels = [str(round(proba, 4)) 
                      if idx % 10 == 0 else ''
                      for idx, proba in enumerate(proba_range)]
            cb.ax.set_yticklabels(labels)
            cb.ax.set_ylabel('Probability')
        ax.add_collection(pc)
            
    NESW = ['N', 'E', 'S', 'W']
    filenames = [df['base_filename'][true_idx] + cardinal_dir + '.png' 
                 for cardinal_dir in NESW]
    if more_options == 'by_probability':
        def add_image(filename, y, label):
            ax_to_add = fig.add_axes([0., y, .32, .20])
            ax_to_add.imshow(imread(filename))
            ax_to_add.set_xticks([]); ax_to_add.set_yticks([])
            ax_to_add.set_ylabel(label)
        y = 0.7
        labels = ['North', 'East', 'South', 'West']
        for filename, label in zip(filenames, labels):
            add_image(filename, y, label)
            y -= 0.2
    # lt, lg = m(-105.5, 39) # test overplot a point
    # m.plot(lt, lg, 'bo', markersize = 24)
    plt.show()

def patch_collection(shape):
    patches = [Polygon(np.array(shape), True)]
    pc = PatchCollection(patches, edgecolor='k', hatch = None, 
                         linewidths=0.5, zorder=2)
    return pc

def plot_this_shapefile(version):
    if version == 'county':
        plot_shapefile('shapefiles/Shape/GU_CountyOrEquivalent', options = 'counties')
    elif version == 'highway':
        plot_shapefile('shapefiles/Highways/SHP/STATEWIDE/HIGHWAYS_latlng', options = 'nofill')
    elif version == 'routes':
        plot_shapefile('shapefiles/Routes/SHP/STATEWIDE/ROUTES_latlng', options = 'nofill')
    elif version == 'majorroads':
        plot_shapefile('shapefiles/MajorRoads/SHP/STATEWIDE/FCROADS_latlng', options = 'nofill')
    elif version == 'localroads':
        plot_shapefile('shapefiles/LocalRoads/SHP/STATEWIDE/LROADS_ogr', options = 'nofill')
    elif version == 'water':
        plot_shapefile('shapefiles/water/watbnd_ogr')
    elif version == 'rocktypes':
        plot_shapefile('shapefiles/COgeol_dd/cogeol_dd_polygon', options = 'rocktypes', cm = 'blues')
    elif version == '12km':
        plot_shapefile('shapefiles/colorado_quadrants/CO_12km_ogr', options = 'other')
    elif version == '24km':
        plot_shapefile('shapefiles/colorado_quadrants/CO_24km_ogr', options = 'other')
    elif version == '100km':
        plot_shapefile('shapefiles/colorado_quadrants/CO_100km_ogr', options = 'other')
    elif version == '1deg':
        plot_shapefile('shapefiles/colorado_quadrants/CO_1deg_ogr', options = 'other')

def convert_shapefile_to_latlng_coord(h_shp):
    ''' Input: Shapefile in projection coordinates. 
        Output: Shapefile in lat/long coordinates for plotting with basemap.
        _latlng will be added to the filename. 
        
        Alternatively (and much more simply):
        ogr2ogr -t_srs EPSG:4326 destination_filename.shp origin_filename.shp

        Careful of the sh.schema.copy(). A shapefile with 3D geometry will
        still crash when using basemap's readshapefile, as it will only 
        accept 2D geometry. Changing the backend (where this would otherwise
        crash) to unpack the 3D tuples for longitude/latitude solves this problem.
        '''
    sh = fiona.open(h_shp)
    orig = Proj(sh.crs)
    dest = Proj(init = 'EPSG:4326')
    with fiona.open(h_shp[:-4] + '_latlng.shp', 'w', 'ESRI Shapefile', 
                    sh.schema.copy(), crs = from_epsg(4326)) as output:
        for feat in sh: 
            points = feat['geometry']['coordinates']
            lat = [point[1] for point in points]
            lng = [point[0] for point in points]
            if any(isinstance(sublist, list) for sublist in points):
                flattened = [pair for combo in points for pair in combo]
                lat = [point[1] for point in flattened]
                lng = [point[0] for point in flattened]
            x, y = transform(orig, dest, lng, lat) 
            out_points = [(ln, la) for ln, la in zip(x, y)]
            feat['geometry']['coordinates'] = out_points
            # if not any(isinstance(sublist, list) for sublist in out_points):
            output.write(feat)

def convert_this_shapefile(version):
    ''' Convert shapefiles using convert_shapefile_to_latlng_coord.
    Everything in a function for clean testing purposes. '''
    if version == 'localroads':
        convert_shapefile_to_latlng_coord('shapefiles/LocalRoads/SHP/STATEWIDE/LROADS.shp')
    elif version == 'water':
        convert_shapefile_to_latlng_coord('shapefiles/water/watbnd.shp')
    elif version == 'majorroads':
        convert_shapefile_to_latlng_coord('shapefiles/MajorRoads/SHP/STATEWIDE/FCROADS.shp')
    elif version == 'routes':
        convert_shapefile_to_latlng_coord('shapefiles/Routes/SHP/STATEWIDE/ROUTES.shp')
    elif version == 'highways':
        convert_shapefile_to_latlng_coord('shapefiles/Highways/SHP/STATEWIDE/HIGHWAYS.shp')

def load_features_and_shape(options):
    if options == 'counties':
        fc = fiona.open("shapefiles/Shape/GU_CountyOrEquivalent.shp")
        feature_name_and_shape = [(fiona_shapefile['properties']['COUNTY_NAM'], 
                                fiona_shapefile['geometry']) 
                                for fiona_shapefile in fc
                                if fiona_shapefile['properties']['STATE_NAME'] == 
                                'Colorado']
    if options == 'geologic_history':
        fc = fiona.open('shapefiles/COgeol_dd/cogeol_dd_polygon.shp')
        feature_name_and_shape = [(fiona_shapefile['properties']['UNIT_AGE'],
                                fiona_shapefile['geometry'])
                                for fiona_shapefile in fc]
    return feature_name_and_shape

# def shape_contains(coord, options = 'counties'):
    # if options == 'counties':
        # fc = fiona.open("shapefiles/Shape/GU_CountyOrEquivalent.shp")

def find_which_feature(coord, feature_name_and_shape):
    for feature_name, feature_shape in feature_name_and_shape:
        sh = sg.asShape(feature_shape)
        if sh.contains(shapely.geometry.Point(coord)):
            return feature_name
        else:
            continue

def load_geologic_history():
    ''' Load the ages of each timeperiod that rocks were formed
    (in millions of years). Used to decrease the number of unique
    categories defining the state. '''
    geologic_times = {None: np.nan,
                    'Cambrian': 500,
                    'Cretaceous': 100,
                    'Cretaceous-Jurassic': 150,
                    'Devonian-Cambrian': 450,
                    'Devonian-Ordivician': 400,
                    'Early Proterozoic': 2300,
                    'Early-Middle Proterozoic': 1600,
                    'Jurassic': 170,
                    'Jurassic-Triassic': 200,
                    'Late Archean': 2800,
                    'Lower Cretaceous-Triassic': 100,
                    'Mesozoic-Pennsylvanian': 300,
                    'Middle Proterozoic': 1400,
                    'Mississipian': 350,
                    'Mississipian-Cambrian': 450,
                    'Mississippian-Cambrian': 450,
                    'Mississippian-Ordovician': 430,
                    'Ordovician': 470,
                    'Ordovician-Cambrian': 490,
                    'Pennsylvanian': 310,
                    'Permian': 270,
                    'Permian-Pennsylvanian': 290,
                    'Quaternary': 1,
                    'Quaternary-Tertiary': 5,
                    'Tertiary': 10,
                    'Tertiary-Cretaceous': 50,
                    'Triassic': 70,
                    'Triassic-Pennsylvanian': 240,
                    'Triassic-Permian': 280}
    return geologic_times
