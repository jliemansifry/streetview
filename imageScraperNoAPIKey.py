import time
import itertools
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import googlemaps
import urllib


'''
This scraper utilizes the website instantstreetview.com to find
valid latitude/longitude coordinates for which there is streetview data.
I played around with querying the streetview API with locations from
road shapefiles in Colorado, but the miss rate was simply too high: it wasn't
worth having 50% of the downloaded images be images that said "Sorry, we have
no imagery here." It's possible to query the streetview API directly in
javascript, but the Python interface doesn't allow this. Utilizing the
middleman of instantstreetview made things easier at the scale of my project,
but this technique wouldn't work on a larger scale.
'''

gmaps_API = googlemaps.Client(key='*********')
geocoder_API = '*******'
streetview_API_key = '********'


def save_image(coord, heading, pitch=5, fov=90, loc='data'):
    '''
    INPUT:  (1) tuple: latitude and longitude coordinates, in degrees
            (2) integer: 0, 360 = N, 90= E, 180 = S, 270 = W
            (3) integer: -90 < pitch < 90 (degrees). 5 is a good middleground
            (4) integer: 20 < fov < 120. 90 is a natural middleground.
            (5) string: folder name to save images to
    OUTPUT: None

    This function will save google street view images facing N, E, S, and W
    for a given coordinate pair to 'loc'
    '''
    if heading == 0 or heading == 360:
        sufx = 'N'
    elif heading == 90:
        sufx = 'E'
    elif heading == 180:
        sufx = 'S'
    elif heading == 270:
        sufx = 'W'
    web_address = ('''https://maps.googleapis.com/maps/api/
                   streetview?size=640x400&location={},{}
                   &fov={}&heading={}&pitch={}&key={}'''.format(
                   coord[0], coord[1], fov,
                   heading, pitch, streetview_API_key))
    filename = ('''{}/lat_{},long_{}_{}.png'''.format(
                loc, str(coord[0])[:8], sufx))
    urllib.urlretrieve(web_address, filename=filename)


def reverse_geocode(coord):
    '''
    INPUT:  (1) tuple: latitude and longitude coordinates, in degrees
    OUTPUT: (1) string: full geocoded address
    '''
    result = gmaps_API.reverse_geocode(coord)
    return result[0]['formatted_address']


def get_elev(coord):
    '''
    INPUT:  (1) tuple: latitude and longitude coordinates, in degrees
    OUTPUT: (1) float: elevation of the lat/lng point in meters
    '''
    elev = gmaps_API.elevation((coord[0], coord[1]))[0]['elevation']
    return elev


def search_colorado():
    '''
    INPUT:  None
    OUTPUT: None

    Search 'Colorado, USA' on instantstreetview.com,
    a website that pulls up valid streetview coordinates.
    '''
    driver.get('http://www.instantstreetview.com/')
    searcher = driver.find_element_by_id("search")
    searcher.click()
    searcher.send_keys("Colorado, USA")
    time.sleep(0.2)
    searcher.send_keys(Keys.RETURN)
    time.sleep(0.5)


def get_random_valid_gps():
    '''
    INPUT:  None
    OUTPUT: string: the web address of a valid google street view location
    '''
    random_button.click()
    time.sleep(1)
    return driver.current_url


def go_to_and_zoom():
    '''
    INPUT:  None
    OUTPUT: None

    Zoom in on Colorado in the sub-map.
    '''
    time.sleep(.3)
    action = webdriver.common.action_chains.ActionChains(driver)
    x_offset_for_zoom_click = -382
    y_offset_for_zoom_click = 209
    action.move_to_element_with_offset(random_button,
                                       x_offset_for_zoom_click,
                                       y_offset_for_zoom_click)
    action.click()
    action.perform()
    time.sleep(1)
    action.perform()


def get_date(coords):
    '''
    INPUT:  None
    OUTPUT: string: the date the image was taken
    '''
    web_url = ('http://maps.google.com/maps?q=&layer=c&cbll={},{}'.format(
               coords[0], coords[1]))
    driver.get(web_url)
    time.sleep(5)
    try:
        titlecard_label = 'widget-titlecard-type-label'
        date = driver.find_element_by_class_name(titlecard_label)
        print date.text
    except:
        timemachine_label = 'widget-timemachine-type-label'
        date = driver.find_element_by_class_name(timemachine_label)
        print date.text
    return date.text


def main():
    '''
    INPUT:  None
    OUTPUT: None

    Do everything. Go to the website, search and zoom in on Colorado, and get
    random valid coordinates for street view locations in Colorado.
    Reverse geocode the valid coordinates to get the elevation and full
    address of the location. Get the date the image was taken.
    Save everything to a .csv.
    '''
    global random_button, driver
    driver = webdriver.Firefox()
    search_colorado()
    time.sleep(3)
    random_button = driver.find_element_by_id('random-button')
    random_button.click()
    time.sleep(5)
    go_to_and_zoom()

    coords = []
    for i in range(200):
        web_url = get_random_valid_gps()
        coord = web_url[34:56].split(',')[:2]
        coords += [(float(coord[0]), float(coord[1]))]
    print coords

    coords_in_colorado = [(lat, lng)
                          for lat, lng in coords
                          if lat > 37 and lat < 41
                          and lng > -109.5 and lng < -102.5]
    print "{} new coords in colorado found".format(len(coords_in_colorado))

    big_list_o_valid_coord = open('big_list_o_trimmed_coords.csv', 'a')
    wr = csv.writer(big_list_o_valid_coord)
    reverse_geocode_info = [reverse_geocode(co) for co in coords_in_colorado]

    dates = [get_date(co)[14:] for co in coords_in_colorado]
    latitudes = [coords_in_colorado[i][0]
                 for i in range(len(coords_in_colorado))]
    longitudes = [coords_in_colorado[i][1]
                  for i in range(len(coords_in_colorado))]
    full_address = [reverse_geocode_info[i]
                    for i in range(len(reverse_geocode_info))]
    elevations = [get_elev(co) for co in coords_in_colorado]

    for row in itertools.izip(latitudes, longitudes,
                              elevations, full_address, dates):
        print row
        if r'\xf1' in full_address or u'\xf1' in full_address:
            pass
        else:
            wr.writerow(row)
    big_list_o_valid_coord.close()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print (end-start)/float(60)
