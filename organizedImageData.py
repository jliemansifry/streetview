import pandas as pd
from imageScraper import save_image
from imagePlaytime import corner_frac, surf
#df = pd.read_csv('big_list_o_updated_coords.csv', names = ['lat', 'lng', 'elev', 'city', 'county', 'state', 'full_address', 'date'])
#df = pd.read_csv('big_list_o_updated_coords.csv') #, names = ['lat', 'lng', 'elev', 'city', 'county', 'state', 'full_address', 'date'])
#df.drop(['city','county','state'], axis=1, inplace = True)
#df.to_csv('big_list_o_trimmed_coords.csv', index = False)

def read_data
df = pd.read_csv('big_list_o_trimmed_coords.csv')


def download_images():
    count = 0
    for lt, lg in zip(df['lat'][6667:7200], df['lng'][6667:7200]): # currently done through image 3500
        # make a catchsafe for overwriting images!!!
        for heading in [0, 90, 180, 270]:
            print count
            count += 1
            save_image((lt,lg), heading)

if __name__ == '__main__':
    pass




#from imageScraper import get_date

#df['date'] = [get_date(co) for co in (zip([lat for lat in df['lat']],[lng for lng in df['lng']]))]
