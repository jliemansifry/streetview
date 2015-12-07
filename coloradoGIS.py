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
# import shapely.geometry as sg


def plot_shapefile(f, options = 'counties', cm = 'blues'):
    num_colors = 29 # 29 unique rock types
    if cm == 'blues':
        cm = plt.get_cmap('Blues')
        blues = [cm(1.*i/num_colors) for i in range(num_colors)]
    else:
        cont_cmap = make_cmyk_greyscale_continuous_cmap()
        blues = [cont_cmap(1.*i/num_colors) for i in range(num_colors)]
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    m = Basemap(width=800000,height=550000, resolution='l',projection='aea', lat_1=37.,lat_2=41,lon_0=-105.55,lat_0=39)
    m.readshapefile(f, name = 'state') # counties
    if options == 'geo':
        rocktypes = np.unique(np.array([m.state_info[i]['ROCKTYPE1'] 
                                        for i in range(len(m.state_info))]))
    for info, shape in zip(m.state_info, m.state):
        if options == 'counties':
            if info['STATE_NAME'] != 'Colorado':
                continue
        if options == 'geo':
            rocktype = info['ROCKTYPE1']
            idx = np.where(rocktypes == rocktype)[0][0]
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches, edgecolor='k', linewidths=.5, zorder=2)
            pc.set_color(blues[idx])
        if options == 'nofill':
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches, edgecolor='k', hatch = None, 
                                linewidths=0.5, zorder=2)
        # else:
            # patches = [Polygon(np.array(shape), True)]
            # pc = PatchCollection(patches, edgecolor='k', linewidths=.5, zorder=2)
            # pc.set_color(random.choice(blues))
        ax.add_collection(pc)
    # lt, lg = m(-105.5, 39) # test overplot a point
    # m.plot(lt, lg, 'bo', markersize = 24)
    plt.show()

# plot_shapefile('shapefiles/Shape/GU_CountyOrEquivalent')
# plot_shapefile('shapefiles/Highways/SHP/STATEWIDE/HIGHWAYS_corr', options = 'nofill')
plot_shapefile('shapefiles/COgeol_dd/cogeol_dd_polygon', 
                options = 'geo', cm = 'ta')

def convert_highways(h_shp):
    sh = fiona.open(h_shp)
    orig = Proj(sh.crs)
    dest = Proj(init = 'EPSG:4326')
    with fiona.open(h_shp + '_corr.shp', 'w', 'ESRI Shapefile', 
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
            if not any(isinstance(sublist, list) for sublist in points):
                output.write(feat)


def load_county_shapefiles():
    fc = fiona.open("shapefiles/Shape/GU_CountyOrEquivalent.shp")
    county_name_and_shape = [(fc[i]['properties']['COUNTY_NAM'], 
                              fc[i]['geometry']) 
                              for i in range(len(fc)) 
                              if fc[i]['properties']['STATE_NAME'] == 'Colorado']
    return fc, county_name_and_shape

def find_which_county(coord, county_name_and_shape):
    for county_name, county_shape in county_name_and_shape:
        sh = shapely.geometry.asShape(county_shape)
        if sh.contains(shapely.geometry.Point(coord)):
            return county_name
        else:
            continue
