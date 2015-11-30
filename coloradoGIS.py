import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shapely
from mpl_toolkits.basemap import Basemap
import shapefile
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
import random
# import shapely.geometry as sg

num_colors = 10 # random colors for testing purposes
cm = plt.get_cmap('Blues')
blues = [cm(1.*i/num_colors) for i in range(num_colors)]

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)

m = Basemap(width=800000,height=550000, resolution='l',projection='aea', lat_1=37.,lat_2=41,lon_0=-105.55,lat_0=39)
m.readshapefile('Shape/GU_CountyOrEquivalent', name = 'state', drawbounds = True) # counties
# m.readshapefile('Shape/GU_IncorporatedPlace', name = 'state', drawbounds = True) # pretty much ignores the mountains; only towns/cities show
for info, shape in zip(m.state_info, m.state):
    if info['STATE_NAME'] != 'Colorado':
        continue
    patches = [Polygon(np.array(shape), True)]
    pc = PatchCollection(patches, edgecolor='k', linewidths=5., zorder=2)
    pc.set_color(random.choice(blues))
    ax.add_collection(pc)

lt, lg = m(-105.5, 39) # test overplot a point
m.plot(lt, lg, 'bo', markersize = 24)
plt.show()


import fiona
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
