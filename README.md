# Google Street View Location Predictor
The emerging field of feature recognition in images is revolutionizing how well computers are able to understand the world around us. Inspired by [geoguessr](https://geoguessr.com/usa/play), my project uses Convolutional Neural Networks (CNN) to discern relevant features that correspond to geographic locations in Colorado. This type of modeling has applications for self-driving cars, where maintaining a keen sense of environment is vitally important. Distinguishing canyon roads from local streets and a clear day from a rainy one will be integral in making smarter autonomous vehicles.

## Table of Contents
1. [The Data](#the-data)
2. [Determining Image Likeness With Vector Based Methods](#determining-image-likeness-with-vector-based-methods)
  * [HSV Color Histograms](#hsv-color-histograms)
  * [Histograms of Oriented Gradients (HOG)](#histograms-of-oriented-gradients)
  * [Speeded-Up Robust Features (SURF)](#speeded-up-robust-features)
  * [Pros and Cons of Vector Based Methods](#pros-and-cons-of-vector-based-methods)
3. [Convolutional Neural Networks](#convolutional-neural-networks)
  * [Training the Neural Nework](#training-the-neural-network)
  * [Results](#results)
  * [Model Gallery](#model-gallery)
  * [iPhone Validation](#iphone-validation)
4. [A Note From the Author](#a-note-from-the-author)

## The Data

15000 locations in Colorado, with google street view images facing North, South, East, and West (60000 total images). The gif below shows all of the locations plotted by Latitude/Longitude/Elevation. The structure of Colorado is quickly apparent; mountains to the West, a high density of roads along the Boulder-Denver-Colorado Springs corridor, and plains to the East.

![Image](/images_for_project_overview/data_animation.gif)

Below are a handful of examples of what the streetview images look like for a given location. The terrain (mountains, plains, etc.) and content (houses, trees, road type, cars, etc.) give clues as to the location in Colorado where each of these images were taken. These hints are easy for humans to pick up on, but difficult for computers.

![Image](/images_for_project_overview/pano_df_idx_5.png)
![Image](/images_for_project_overview/pano_df_idx_25.png)
![Image](/images_for_project_overview/pano_df_idx_340.png)
![Image](/images_for_project_overview/pano_df_idx_3100.png)

## Determining Image Likeness With Vector Based Methods

Describing features in images is possible with a multitude of vector based methods. I explored the use of HSV Color Histograms, Histograms of Oriented Gradients (HOG; Dalal, N. & Triggs, B., 2005), and Speeded-Up Robust Features (SURF; Bay, H., Tuytelaars, T., & Van Gool, L., 2006) for this project, but it has been quite some time since these methods were considered cutting edge, and each has significant limitations. CNNs mimic and surpass the abilities of these methods, but it was worth trying them out as a baseline. More on my implementation of CNN later. 

### HSV Color Histograms

Using a 3D color histogram allows us to describe the color density of various colors in our image. Rather than an RGB (red, green, blue) histogram, I analyzed the color distribution in HSV (hue, saturation, value) space, which represents colors in a way more similar to how humans perceive colors. The RGB color space is represented by a cube (linear additions and subtractions of each channel to form the output), whereas the HSV color space takes RGB intensities and maps them onto a cylinder. See images below. 

![Image](/images_for_project_overview/RGB_cube.png) ![Image](/images_for_project_overview/HSV_cylinder.png)

The number of pixels that fall into a each HSV bin give a quantitative way of describing the color distribution of an image without needing to look at every pixel. A 960 component vector is created by sampling 4 regions in each image, with 8 hue bins, 8 saturation bins, and 3 value bins, and we are able to quantify the likeness between two (normalized) vectors using Euclidean distance. It is significantly easier to compare a vector of this length than one with 768,000 elements, which is what it would be if we were comparing each pixel between images (640pix by 400 pix by 3 colors). This process doesn't sacrifice much detail. In fact, it helps generalize the process, as binning the pixel intensities for each region allows us to compare the color distribution of a given *region* rather than by individual *pixels* between images. Two pictures now need not be nearly identical in order to have a high likeness; rather, only the color distribution of each region needs to be similar. 

![Image](/images_for_project_overview/HSV_greener_usedincomplat_40.35530,long_-103.434_W .jpeg) ![Image](/images_for_project_overview/HSV_redder_usedincomp_lat_39.47983,long_-104.558_S.jpeg)

Pixels from the image on the left (circles) and on the right (diamonds) are plotted below in HSV color space. This isn't exactly the HSV histogram described above (for which the bin counts would be difficult to visualize), but it gives us a sense of the color distribution in HSV space. The sky portion of each image is very similar, filling out the blue region of the HSV space (the histogram bins in this portion of the cylinder would have similar counts), but the green and red ground colors occupy distinctly different regions in the cylinder. 

![Image](/images_for_project_overview/HSV_image_comparison.gif)

Taking the images with the smallest euclidean distance between their color histogram vectors gives exciting results for determining where in Colorado a series of North, East, South, and West images was taken, as seen below. In each case presented, the first panorama acts as the 'search' set of images, and the second set are the most similar set of images found in the database. Often, the most similar set of images is remarkably close in location to the search images, as can be seen in the figure that follows the panoramas. This figure displays the true location of the 'search' panorama, the best guess based on most similar image in the database, and the rest of the nearest 10 locations.

![Image](/images_for_project_overview/pano_likeness_mountains.png)
![Image](/images_for_project_overview/pano_likeness_highway.png) 
![Image](/images_for_project_overview/pano_likeness_grasslands.png) 
![Image](/images_for_project_overview/pano_likeness_all_3.png) 

### Histograms of Oriented Gradients
We can get a sense of the texture and simple features that make up an image by using an implementation of HOG, which calcuates the intensity gradient across a specified pixel region. The strength and orientation of these gradients can then be compared between images to quantify their similarity. Below is an example of HOG in action. 
![Image](/images_for_project_overview/raw_img_vs_hog_img_rotate.png)

### Speeded Up Robust Features
SURF also ascertains features in images by looking at intensity gradients. However, rather than looking at the entire image, the algorithm looks for regions of the highest intensity gradients, their scale, and their orientation. These regions are marked and vectorized as features, which can then be compared between images. In the examples below, each circle denotes a feature, with the line from the center denoting its orientation.
![Image](/images_for_project_overview/surf_examples.png)

### Pros and Cons of Vector Based Methods
Vector based methods can work remarkably well at matching images within a dataset. However, there are a series of stipulations to fulfill this *can*. There are enough images in this dataset that it is not uncommon for there to be pairs (or more) of images taken with the same google street view car on the same road during the same season at the same time of day. That's a lot of sames. Without these conditions, this technique simply wouldn't work. Change any one of these things and the similarity of the images, as determined by the computer, would plummet. 

Additionally, vector based methods are not translationally invariant at their core. If two otherwise quite similar images were shifted from one another, the computer would no longer see them as being so similar. While possible to account for transformations of various kinds (rotation, translation, reflection, scaling, etc.) when comparing images, this would drive the computational requirements through the roof for a dataset of this size. It simply isn't practical when you need to compare your 'search' image to thousands of other images. 

## Convolutional Neural Networks 
Deep Convolutional Neural Networks (DCNNs) have proven to be the most flexible and powerful tool for image recognition, achieving near-human performance on a variety of classification tasks. Their ability to recognize abstract features that distinguish classes, translational invariance by pooling, and capacity to exclude unreasonable patterns given enough data are just a few of many reasons that DCNNs significantly outperform vector based methods. Utilizing a DCNN to predict location in Colorado from the google street view data was a logical step after recognizing the limitations of the vector based methods previously explored.

### Training the Neural Network
I trained my DCNN using Keras and Theano on an AWS GPU-optimized machine complete with CUDA. In building the net, I initially started by mimicking the structure of a few past winners of the ImageNet classification competition (Krizhevsky, A., et al, 2012; He, K., et al, 2015). However, their nets were designed to classify 1.2 million images of size 224x224 pixels into one of 1,000 classes, whereas I was working with 80,000 images of size 80x50 and trying to classify into one of 64 classes. This basic difference meant that not much of their structure mapped onto my problem. For example, Krizhevsky et al. used 11x11 convolutions with stride 4 in the first layer, dimensionality significantly out of scale for my images.

Although the images were scraped at a resolution of 640x400 pixels, the time needed to train even the most basic of models was simply too great for me to use these full resolution images in the span of this 3 week Galvanize capstone project. In downsizing them to 80x50, I certainly forfeited some distinguishing details, but gained the ability to iterate through a multitude of model structures and ensure I was squeezing every last drop of information out of the images. A thoroughly trained model on low resolution data will almost always outperform an ill-trained model utilizing high resolution images.

A cartoon of the final model structure used is below. Images taken facing in each cardinal direction (20,000 apiece) were fed into the network separately, with each pipeline having unique kernels and weights. After many iterations, I found that having 2 pipelines for each cardinal direction, with differing initial convolutional sizes (3x3 and 5x5), resulted in a significant jump in validation accuracy, holding everything else constant (from 0.18 to 0.23). Adjusting the number of nodes, dropout, and utilizing advanced activations (Leaky ReLU and Parametric ReLU) further raised the validation accuracy to 0.38 on the same test set. 

![Image](/images_for_project_overview/model_architecture.png)

### Results

Below are a few examples of how well the net is doing. The images on the left act as the 'search' images and come from the shaded county. Here they are presented in full 640x400 resolution, though in reality the net was fed the 80x50 versions of these images. The colors correspond to the predicted probabilities (according to the CNN) of the images coming from each of the counties of Colorado. 

In each of the examples, the net predicts the counties where the features detected in the 'search' images best match the learned features from seeing hundreds of google street view images during training. 

In the first example, the net correctly predicts Summit county, but neighboring Eagle county also receives a high probability, as the net has learned it also shares the snow/trees/mountain features that Summit county has. With the second set of images, clearly suburban, the net correctly predicts Jefferson county, and also assigns a non-zero probability to all the other counties in Colorado that have some measure of suburban sprawl. The net incorrectly guesses Mesa county when the true county was Garfield in the third example, but is clearly in the correct geographic region of Colorado. In the fourth example, there are few distinguishing features, other than perhaps *not mountains*. Sure enough, the net recognizes that the images are not from the mountains, but isn't sure where in the plains the images are from either (not that a human could do much better). 

![Image](/images_for_project_overview/county_model_v1.2_Summit_idx_203.png)
![Image](/images_for_project_overview/county_model_v1.2_Jefferson_idx_156.png)
![Image](/images_for_project_overview/county_model_v1.2_Garfield_idx_202.png)
![Image](/images_for_project_overview/county_model_v1.2_El Paso_idx_166.png)

### Model Gallery
See [here](/model_gallery.md) for more examples of how the net performed on a variety of input landscapes and counties. 

### iPhone Validation

Out of curiosity, I went out to Platte Street in Denver (right outside Galvanize) and took photos facing North, South, East, and West with my iPhone. Sure enough, the net correctly identified these images as being from Denver! 

![Image](/images_for_project_overview/county_model_v1.2_Denver.png)

It would be great to continue assessing this model with more images not from google street view in order to see how it performs in various circumstances. It is worth noting that the net was trained with images always taken on a road, so images from other situations (ie. a hike) wouldn't perform as well. Given more time, I would have loved to include geotagged images from Flickr, Imgur, etc., in order to improve performance. 

## A Note From the Author

Questions? Comments? Want to try some transfer learning with my trained model? Want to get your hands on all the images I used? Contact me at jliemansifry@gmail.com! 
