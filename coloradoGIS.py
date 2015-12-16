import pandas as pd
from mpl_toolkits.basemap import Basemap
from pyproj import Proj, transform
import fiona
from fiona.crs import from_epsg
import numpy as np
import matplotlib.pyplot as plt
import shapely
import shapefile
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection, LineCollection
from imagePresentationFunctions import make_cmyk_greyscale_continuous_cmap
import random
import shapely.geometry as sg

def plot_shapefile(f, options = 'counties', cm = 'blues', df = None):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    m = Basemap(width=800000,height=550000, resolution='l', projection='aea',
                lat_1=37.,lat_2=41,lon_0=-105.55,lat_0=39)
    m.readshapefile(f, name = 'state')
    if options == 'rocktypes':
        rocks= np.unique([shape_info['ROCKTYPE1']
                          for shape_info in m.state_info])
    if options == 'geologic_history':
        geologic_time_dictionary = load_geologic_history()
        ranges = [0, 5, 20, 250, 500, 3000] # in Myr
        all_ranges = []
        for r, r_plus1 in zip(ranges[:-1], ranges[1:]):
            all_ranges += [str(r) + '-' + str(r_plus1)]
        rocks = all_ranges
    num_colors = 10 # len(rocks)
    if cm == 'blues':
        cm = plt.get_cmap('Blues')
        blues = [cm(1.*i/num_colors) for i in range(num_colors)]
    else:
        cont_cmap = make_cmyk_greyscale_continuous_cmap()
        blues = [cont_cmap(1.*i/num_colors) for i in range(num_colors)]
    for info, shape in zip(m.state_info, m.state):
        if options == 'counties':
            if info['STATE_NAME'] != 'Colorado':
                continue
            county_name = info['COUNTY_NAM']
            print county_name
            locs = np.where(df['county'] == county_name)[0]
            r_avg = df['avg_r_low'][locs].mean()
            g_avg = df['avg_g_low'][locs].mean()
            b_avg = df['avg_b_low'][locs].mean()
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches, edgecolor='k', hatch = None, 
                                 linewidths=0.5, zorder=2)

            # sh = sg.asShape(np.array(shape))
            # r_avg, g_avg, b_avg= [], [], []
            # for idx in range(10): #range(df.shape[0]):
                # print idx
                # coord = zip(df['lng'][idx], df['lat'][idx])
                # if sh.contains(shapely.geometry.Point(coord)):
                # r_avg += [df['avg_r_low'][idx] / 255.]
                # g_avg += [df['avg_g_low'][idx] / 255.]
                # b_avg += [df['avg_b_low'][idx] / 255.]
                # if r_avg == []:
                    # r_avg, g_avg, b_avg = 0., 0., 0.
            # pc.set_color(random.choice(blues))
            pc.set_color((r_avg/255., g_avg/255., b_avg/255.))
        elif options == 'rocktypes':
            rocktype = info['ROCKTYPE1']
            idx = np.where(rocks == rocktype)[0][0]
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches, edgecolor='k', linewidths=.5, 
                                 zorder=2)
            pc.set_color(blues[idx])
        elif options == 'geologic_history':
            rock_age = info['UNIT_AGE']
            for index, (r, r_plus1) in enumerate(zip(ranges[:-1], ranges[1:])):
                try:
                    if ((geologic_time_dictionary[rock_age] > r) & 
                    (geologic_time_dictionary[rock_age] < r_plus1)):
                        idx = index 
                except: 
                    idx = 2 #20-250 was a good middleground for nans
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches, edgecolor='k', linewidths=.5, 
                                zorder=2)
            pc.set_color(blues[idx])
        elif options == 'nofill':
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches, edgecolor='k', hatch = None, 
                                 linewidths=0.5, zorder=2)
            pc.set_facecolor('none')
        else:
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches, edgecolor='k', hatch = None, 
                                 linewidths=0.5, zorder=2)
            pc.set_color(random.choice(blues))
        ax.add_collection(pc)
    # lt, lg = m(-105.5, 39) # test overplot a point
    # m.plot(lt, lg, 'bo', markersize = 24)
    plt.show()

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
