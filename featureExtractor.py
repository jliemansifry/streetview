from imageProcessor import ColorDescriptor
import argparse
import glob
import cv2

dataset_dir = "data"

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required = True,
            # help = "Path to the directory that contains the images to be indexed")
# ap.add_argument("-i", "--index", required = True,
            # help = "Path to where the computed index will be stored")
# args = vars(ap.parse_args())
 
# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

# open the output index file for writing
# output = open(args["index"], "w")
 
# use glob to grab the image paths and loop over them

# for image_path in glob.glob(args["dataset"] + "/*.png"):
    # for lat, lng in zip(image_path[22:30], image_path[36:44]):
        # image_filename = image_path[image_path.rfind("/") + 1:]
import pandas as pd
df = pd.read_csv('big_list_o_trimmed_coords.csv')
NESW = ['N', 'E', 'S', 'W']
for lt, lg in (df['lat'][:1000], df['lng'][:1000]):
    ltlg_image_path = dataset_dir + '/lat_' + lt + ',long_' + lg
    for cardinal_dir in NESW:
        # image_path = args["dataset"] + '/lat_' + lt + ',long_' + lg + '_' + cardinal_dir + '.png'
        image = cv2.imread(ltlg_image_path + '_' + cardinal_dir + '.png')
        ltlg_features = [cd.describe(image)]
        #does this work???????????
        df['hist_vec_' + cardinal_dir] = ltlg_features

# df = pandas.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
# for i in range(100):
        # df.iloc[i]['E'] =df.iloc[i]['A'] +df.iloc[i]['B']

    # WRITE TO YOUR DF
    # features = [str(f) for f in ltlg_features]
    # output.write("%s,%s\n" % (ltlg_image_path, ",".join(features)))
# for image_path in glob.glob(args["dataset"] + "/*.png"):
    # # extract the image ID (i.e. the unique filename) from the image
    # # path and load the image itself
    # image_filename = image_path[image_path.rfind("/") + 1:]
    # image = cv2.imread(image_path)
 
    # # describe the image
    # features = cd.describe(image)
    # # write the features to file
    # features = [str(f) for f in features]
    # output.write("%s,%s\n" % (image_filename, ",".join(features)))
# output.close()

# on bash: python index.py --dataset dataset --index index.csv
