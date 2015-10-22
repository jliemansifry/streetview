import pandas as pd

#df = pd.read_csv('big_list_o_updated_coords.csv', names = ['lat', 'lng', 'elev', 'city', 'county', 'state', 'full_address', 'date'])
#df = pd.read_csv('big_list_o_updated_coords.csv') #, names = ['lat', 'lng', 'elev', 'city', 'county', 'state', 'full_address', 'date'])
#df.drop(['city','county','state'], axis=1, inplace = True)
#df.to_csv('big_list_o_trimmed_coords.csv', index = False)
df = pd.read_csv('big_list_o_trimmed_coords.csv')

df['lat'][:10]
df['lng'][:10]

from imageScraper import save_image
count = 0
for lt, lg in zip(df['lat'][1000:1100], df['lng'][1000:1100]): # currently done through image 500
    # make a catchsafe for overwriting images!!!
    for heading in [0, 90, 180, 270]:
        print count
        count += 1
        save_image((lt,lg), heading)

#from imageScraper import get_date
 
#df['date'] = [get_date(co) for co in (zip([lat for lat in df['lat']],[lng for lng in df['lng']]))]
