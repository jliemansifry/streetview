Google Street View Locaiton Predictor
The emerging field of feature recognition in images is revolutionizing how well computers are able to understand the world around us. Inspired by geoguessr, my project uses convolutional neural networks to discern relevant features that correspond to geographic locations in Colorado. This type of modeling has applications for self-driving cars, where maintaining a keen sense of environment is vitally important. Distinguishing canyon roads from local streets and a clear day from a rainy one will be integral in making smarter autonomous vehicles.

The Data

15000 locations in Colorado, with google street view images facing North, South, East, and West (60000 total images). The gif below shows all of the locations plotted by Latitude/Longitude/Elevation. The structure of Colorado is quickly apparent; mountains to the West, a high density of roads along the Boulder-Denver-Colorado Springs corridor, and plains to the East.

![Image](/images_for_project_overview/data_animation_gif.gif)

Below are a handful of examples of what the streetview images look like for a given location. The terrain (mountains, plains, etc.) and content (houses, trees, road type, cars, etc.) give clues as to the location in Colorado where each of these images were taken. These hints are easy for humans to pick up on, but difficult for computers.

![Image](/images_for_project_overview/data_animation_gif.gif)
![Image](/images_for_project_overview/pano_df_idx_5.png)
![Image](/images_for_project_overview/pano_df_idx_25.png)
![Image](/images_for_project_overview/pano_df_idx_150.png)
![Image](/images_for_project_overview/pano_df_idx_250.png)
![Image](/images_for_project_overview/pano_df_idx_340.png)
![Image](/images_for_project_overview/pano_df_idx_3100.png)
