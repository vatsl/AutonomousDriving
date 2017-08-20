# Vehicle Detection using Support Vector Machine and HOG Features

**Source Code: Vehicle_Detection.ipynb**

The aim is to develop a pipeline which tracks the cars detected along the lanes.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: examples/car_not_car.png
[image2]: examples/HOG_example.jpg
[image3]: examples/sliding_windows.jpg
[image4]: examples/output_bboxes.png
[image5]: examples/heatmap.jpg
[image8]: examples/hog_images_nocar.png

---

### Histogram of Oriented Gradients (HOG)

#### Extracting HOG features from the training images

I used the *hog()* function from the skimage.feature package to extract the hog features. 

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. The plots are shown below.

The hog features are extracted individually from each of the color channels.

The code for this step is contained in the function *get_hog_features* of the IPython notebook. After a lot of trial runs, I settled on the following values of parameters: orient=9; pix_per_cell=12, cell_per_block=8, color_space=RGB.

An example of HOG features is given below.
![alt text][image2]


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

#### Training a classifier using the selected HOG features and color features.

I have trained my classifier using SVC function sklearn.svm. SVC was chosen instead of LinearSVC because it resulted in increased accuracy. The increase in accuracy was because of the probabilistic output of SVC.

The implementations are present in the funtion *svm_classifier()* in the attached IPYNB notebook.
 
I tried with different set of features and trained my classifier several times with them. I tried with HSV, HLS and RGB color spaces with orient value of 9, 12 and different pix_per_cell values. Each training cycle would take around 30-45 minutes on my machine and then further application to the video file would take more than 4-5 hours. This meant I was leaving my machine running overnight to get a result.

Finally I went with the parameters described in the earlier section. My current model gave me an accuracy of 99.53%.

### Sliding Window Search

There are some helper functions for sliding window search present in the attaached notebook namely, *draw_boxes()*, *slide_window()*, *add_heat()*, *apply_threshold()*, *draw_labeled_bboxes()*. The ipynb file mentions their descriptions as well.
  
The ***image_pipeline()*** function contains the whole implementation. It applies all the defined functions to an image.
  
Coming up with parameters involved extensive trail and error. Also, there is no fixed single answer. Different configurations can be used to build the pipeline.

Three different window sizes were selected. 
      
For small windows :-
     
          a) Window Size - (64, 64)
          b) X Start Stop - (640, 1280)
          c) Y Start Stop - (400, 700)
          d) Overlap - (0.6, 0.6)
          
For medium windows :-
     
          a) Window Size - (96, 96)
          b) X Start Stop - (640, 1280)
          c) Y Start Stop - (400, 700)
          d) Overlap - (0.75, 0.75)
          
For large windows :-
     
          a) Window Size - (128, 128)
          b) X Start Stop - (640, 1280)
          c) Y Start Stop - (400, 700)
          d) Overlap - (0.8, 0.8)
          
All the windows obtained from above step are then passed to *search_windows()* to get the actual hot boxes using prediction from SVC classifier. The classifier is used to predict the probability of a car being present in the image. If probability is greater than 0.5 then, car is present in the image.
  
Then a heatmap is created using the hot boxes from above step. *add_heat()* function is used. +1 value is added to all the boxes obtained from above step. After that *apply_threshold()* function is called to remove some of the false positives present.
  
*label()* function from *scipy.ndimage.measurements* helps in labelling the thresholding regions from the above step. After that *draw_labeled_bboxes()* function is called to draw bounding box around the detected car.
 
Example of final output from the pipeline:

Sliding windows before applying heatmaps
![alt text][image3]

After applying heatmaps
![alt text][image5]

Final output from pipeline
![alt text][image4]

To remove False positives :-

In *draw_labeled_bboxes()* function, I check whether the minimum y-coordinate values is in between 300 and 600. If not, then don't draw the box. Also, minimum x-coordinate value should be less than 1220. One more check of the area of the box has been used to remove small False positve boxes. If the area of the box is less than 2500, then don't draw it.

Use of SVC with probabilitic prediction instead of normal LinearSVC also helped in removing many False positives. 

Heatmaps and thresholding on those heatmaps also helped in removing False positives.

---

### Video Implementation

The output video is available on [YouTube](https://youtu.be/Y3_8yzNPso4) and in the repositury as **p5_result.mp4**

---

#### Filter for false positives and some method for combining overlapping bounding boxes

Method used for removing False positive has been discussed earlier.

Bounding boxes obtained in image pipeline in each frame were flickering a lot. So, to reduce the flickering Exponential Smoothening  or a first class filtering has been used. It's implementation is present in inside *image_pipeline()* function. Heatmaps obtained from previous frame is given weightage of **0.2** and heatmaps from current frame are given weightage of **0.8**. This is the way in which we decide the heatmap for current frame. These two values were obtained after a few experiments.


---

### Reflections

- The pipeline used in the current project is very specific to the project video. All the different parameters have been tuned keeping that fact in mind. The training image set is hardly capable of working on a generalized model and I don't think HOG + SVM is sufficiently capable enough to detect cars well. There is a need for generlistic approach.


**Making it robust**

- Adding more diverse training data.
- Move to Deep Learning using GPUs (this should significantly increase speed as well)
- Reduce the search space for the sliding window by searching only at the entry and exit points beyond the already found cars. This way we will have only a very limited set of sliding windows to search.
- Another idea is, once a car is "locked", we can use simple template matching to keep finding that car and need not use a full sliding window approach.
- Try other features such as Haar, MOG and see their relative performance.
