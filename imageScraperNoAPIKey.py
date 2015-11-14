import time
import itertools
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import googlemaps
import random
import urllib

gmaps_API = googlemaps.Client(key='*********')
geocoder_API = '*******'
streetview_API_key = '********'

CO_NW = (41.0,-109.05)
CO_SW = (37.0,-109.05)
CO_SE = (37.0,-102.05)
CO_NE = (41.0,-102.05)
def gen_random_coord(NW, SW, SE, NE): #will work for square states but not all
    coord = (random.uniform(NW[0],SW[0]),random.uniform(NW[1],NE[1]))
    return coord

def save_image(coord, heading, pitch = 5, fov = 90):
    ''' heading of 0,360 = N, 90= E, 180 = S, 270 = W '''
    if heading == 0 or heading == 360:
        sufx = 'N'
    elif heading == 90:
        sufx = 'E'
    elif heading == 180:
        sufx = 'S'
    elif heading == 270:
        sufx = 'W'
    web_address = 'https://maps.googleapis.com/maps/api/streetview?size=640x400&location='+str(coord[0])+','+str(coord[1])+'&fov='+str(fov)+'&heading='+str(heading)+'&pitch='+str(pitch)+'&key='+streetview_API_key
    filename = 'data/lat_'+str(coord[0])[:8]+',long_'+str(coord[1])[:8]+'_'+sufx+'.png'
    urllib.urlretrieve(web_address, filename = filename)

def reverse_geocode(coord): #returns city, state, full address
    result = gmaps_API.reverse_geocode(coord)
    return result[0]['formatted_address']

def get_elev(coord):
    elev = gmaps_API.elevation((coord[0],coord[1]))[0]['elevation'] # needs a tuple
    return elev

# driver = webdriver.Firefox()

def search_colorado():
    driver.get('http://www.instantstreetview.com/')
    searcher = driver.find_element_by_id("search")
    searcher.click()
    searcher.send_keys("Colorado, USA")
    time.sleep(0.2)
    searcher.send_keys(Keys.RETURN)
    time.sleep(0.5)
    
def get_random_valid_gps():
    random_button.click()
    time.sleep(1)
    return driver.current_url

def go_to_and_zoom():
    time.sleep(.3)
    action = webdriver.common.action_chains.ActionChains(driver)
    action.move_to_element_with_offset(random_button,-382,209) #400 left, 200 down. from random button
    action.click()
    action.perform()
    time.sleep(1)
    action.perform()

def get_date(coords):
    web_url = 'http://maps.google.com/maps?q=&layer=c&cbll='+str(coords[0])+','+str(coords[1])
    driver.get(web_url)
    time.sleep(5)
    try:
        date = driver.find_element_by_class_name('widget-titlecard-type-label')
        print date.text
    except:
        date = driver.find_element_by_class_name('widget-timemachine-type-label')
        print date.text
    return date.text

def main():
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
        coords += [(float(coord[0]),float(coord[1]))]
    print coords

    coords_in_colorado = [(lat,lng) for lat,lng in coords if lat > 37 and lat < 41 and lng > -109.5 and lng < -102.5]
    print "{} new coords in colorado found".format(len(coords_in_colorado))
    
    big_list_o_valid_coord = open('big_list_o_trimmed_coords.csv','a')
    wr = csv.writer(big_list_o_valid_coord)
    reverse_geocode_info = [reverse_geocode(co) for co in coords_in_colorado]

    dates = [get_date(co)[14:] for co in coords_in_colorado]
    latitudes = [coords_in_colorado[i][0] for i in range(len(coords_in_colorado))]
    longitudes = [coords_in_colorado[i][1] for i in range(len(coords_in_colorado))]
    full_address = [reverse_geocode_info[i] for i in range(len(reverse_geocode_info))]
    elevations = [get_elev(co) for co in coords_in_colorado]

    for row in itertools.izip(latitudes, longitudes, elevations, full_address, dates):
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
