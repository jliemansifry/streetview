import pandas as pd
from pyproj import Proj, transform
import fiona
from fiona.crs import from_epsg
import numpy as np
import matplotlib.pyplot as plt
import shapely
from mpl_toolkits.basemap import Basemap
import shapefile
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
import random
# import shapely.geometry as sg

num_colors = 29 # random colors for testing purposes
cm = plt.get_cmap('Blues')
blues = [cm(1.*i/num_colors) for i in range(num_colors)]

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)

m = Basemap(width=800000,height=550000, resolution='l',projection='aea', lat_1=37.,lat_2=41,lon_0=-105.55,lat_0=39)

def plot_shapefile(f, options = 'counties'):
    m.readshapefile(f, name = 'state') # counties
    if options == 'geo':
        rocktypes = np.unique(np.array([m.state_info[i]['ROCKTYPE1'] for i in range(len(m.state_info))]))

    for info, shape in zip(m.state_info, m.state):
        if options == 'counties':
            if info['STATE_NAME'] != 'Colorado':
                continue
        if options == 'geo':
            rocktype = info['ROCKTYPE1']
            idx = np.where(rocktypes == rocktype)[0][0]
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches, edgecolor=None, facecolor = blues[idx], linewidths=.5, zorder=2)
        else:
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches, edgecolor='k', linewidths=5., zorder=2)
            pc.set_color(random.choice(blues))
        ax.add_collection(pc)
    # lt, lg = m(-105.5, 39) # test overplot a point
    # m.plot(lt, lg, 'bo', markersize = 24)
    plt.show()

def convert_highways(h_shp):
    sh = fiona.open(h_shp)
    orig = Proj(sh.crs)
    dest = Proj(init = 'EPSG:4326')
    with fiona.open('Highways/SHP/STATEWIDE/HIGHWAYS_corr.shp', 'w', 'ESRI Shapefile', sh.schema.copy(), crs = from_epsg(4326)) as output:
        for feat in sh: # feat = one polygon of the shapefile
            out_linearRing = [] # empty list for the LinearRing of transformed coordinates
            for point in feat['geometry']['coordinates']: # LinearRing of the Polygon
                print point
                lng,lat =  point  # one point of the LinearRing
                x,y = transform(orig, dest,lng,lat) # transform the point
                print x,y 
                out_linearRing.append((x,y)) # add all the points to the new LinearRing
            # transform the resulting LinearRing to  a Polygon and write it
            feat['geometry']['coordinates'] = [out_linearRing]
            output.write(feat)

# plot_shapefile('Shape/GU_CountyOrEquivalent')
# plot_shapefile('Highways/SHP/STATEWIDE/HIGHWAYS')
# plot_shapefile('COgeol_dd/cogeol_dd_polygon', options = 'geo')


fc = fiona.open("Shape/GU_CountyOrEquivalent.shp")
# county_names = [fc[i]['properties']['COUNTY_NAM'] for i in range(len(fc)) if fc[i]['properties']['STATE_NAME'] == 'Colorado']
county_name_and_shape = [(fc[i]['properties']['COUNTY_NAM'], fc[i]['geometry']) for i in range(len(fc)) if fc[i]['properties']['STATE_NAME'] == 'Colorado']

def find_which_county(coord):
    for county_name, county_shape in county_name_and_shape:
        sh = shapely.geometry.asShape(county_shape)
        if sh.contains(shapely.geometry.Point(coord)):
            return county_name
        else:
            continue
