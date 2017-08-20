[//]: # (Image References)
[loading_screen]: ./resources/loading_screen.png
[distortion]: ./resources/output_images/distortion.png
[distortion_theory]: ./resources/distortion.png
[corners_unwarp]: ./resources/output_images/corners_unwarp.png
[distortion_corrected]: ./resources/output_images/undistorted.png
[sobel_x]: ./resources/output_images/sobel_x.png
[sobel_y]: ./resources/output_images/sobel_y.png
[gradient_magnitude]: ./resources/output_images/gradient_magnitude.png
[gradient_direction]: ./resources/output_images/gradient_direction.png 
[color_thresholds]: ./resources/output_images/color_thresholds.png 
[multi_thresholds]: ./resources/output_images/thresholded_binary.png
[region_masked]: ./resources/output_images/region_masked.png
[perspective_transform]: ./resources/output_images/perspective_transform.png
[sliding_windows]: ./resources/output_images/sliding_windows.png
[shaded_lanes]: ./resources/output_images/shaded_lanes.png
[lane_mapping]: ./resources/output_images/lane_mapping.png
[test_image1]: ./resources/test_images/test1.jpg
[test_image2]: ./resources/test_images/test2.jpg
[test_image3]: ./resources/test_images/test3.jpg
[test_image4]: ./resources/test_images/test4.jpg
[test_image5]: ./resources/test_images/test5.jpg
[test_image6]: ./resources/test_images/test6.jpg

# Advanced Lane Finding

**Source Code: Advanced_Lane_Detection.ipynb**

This project utilizes several computer vision algorithms and techniques to perform advanced lane finding on test images and video streams. There following steps are involved: 

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Apply a distortion correction to raw images
* Use color transforms and gradient thresholding to create a thresholded binary image
* Apply a perspective transform to get a bird's eye view of the image
* Detect lane pixels using sliding window method and convolutions
* Determine the lane curvature and vehicle position with respect to center
* Warp the detected lane boundaries back onto the original image. Print the lane information on the image


### Camera Calibration

The code for this step is contained in the attached P4.ipynb file, more specifically in the Section 1.  

![alt text][distortion]

First, we define "object points", which represent the (x, y, z) coordinates of the chessboard corners in the world.We assume that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` is appended with a copy of it every time we successfully detect all chessboard corners in a test image.  `imgpoints` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][distortion_theory]

From there, we use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.We apply this distortion correction to the test image using the `cv2.undistort()` function and obtain this result: 

![alt text][corners_unwarp]

Here are the results of distortion correction on each of the test images (located in the test_images folder):

![alt text][distortion_corrected]

---

### Image Processing

Section 2 of the code explains this process step-by-step, with examples of each individual threshold. In order, the thresholding used on the images is as follows:

+ Color Channel HLS & HSV Thresholding - Extract the S-channel of the original image in HLS format and combine the result with the extracted V-channel of the original image in HSV format.

![alt text][color_thresholds]

+ Binary X & Y - I use Sobel operators to filter the original image for the strongest gradients in both the x-direction and the y-direction.

![alt text][sobel_x]
![alt text][sobel_y]

Next, using techniques from "Finding Lane Lines Project", conduct sanity checks. These techniques included Region Masking & Hough Lines, and the purpose for performing them was to ensure that the thresholding steps taken are accurate enough to yield proper perspective transforms.

The region masking results are shown below. Region is masked with the following points: 

| Point       | Value                                    | 
|:-----------:|:----------------------------------------:| 
| Upper Left  | (image width x 0.4, image height x 0.65) | 
| Upper Right | (image width x 0.6, image height x 0.65) |
| Lower Right | (image width, image height)              |
| Lower Left  | (0, image height)                        |

![alt text][region_masked]

The code for perspective transform is performed in a function called perspective_transform. The function takes in a thresholded binary image and source points, with the source points coinciding with the region masking points explained in the region masking table above. For destination points, the outline of the image being transformed is chosen. Here are the results of the transforms:

![alt text][perspective_transform]

The next step after transforming the perspective was to detect lane-line pixels and to fit their positions using a polynomial in Section 4 of the code. After developing functions for sliding_windows and shaded_lanes, we are able to detect the lanes and yield the following results:

Sliding Windows Technique:
![alt text][sliding_windows]

Shaded Lanes Technique:
![alt text][shaded_lanes]

After detecting the lanes, calculate the radius of curvature for each of the polynomial fits. The results of these calculations are shown in the table below. Use the radius of curvature example code from Udacity's lessons to create the calculation cells.

| Test Image                | Radius of Curvature (Left) | Radius of Curvature (Right) | 
|:-------------------------:|:--------------------------:|:---------------------------:| 
| ![alt-text][test_image1]  | 3101.953248 meters         | 1373.981912 meters          | 
| ![alt-text][test_image2]  | 2227.880643 meters         | 1447.535637 meters          |
| ![alt-text][test_image3]  | 10088.08471 meters        | 1469.011088 meters          |
| ![alt-text][test_image4]  | 330.567254 meters          | 1382.172793 meters          |
| ![alt-text][test_image5]  | 381.390857 meters          | 1868.689683 meters          |
| ![alt-text][test_image6]  | 411707.661 meters          | 1470.278540 meters          |

Another calculation performed was the offset from the lane's center. The calculations are shown in the code cell following the radius of curvature, and yielded the following:

| Test Image                | Offset from Center |
|:-------------------------:|:------------------:| 
| ![alt-text][test_image1]  | -0.070 meters      |
| ![alt-text][test_image2]  | -0.063 meters      |
| ![alt-text][test_image3]  | -0.168 meters      |
| ![alt-text][test_image4]  | -0.192 meters      |
| ![alt-text][test_image5]  | -0.255 meters      |
| ![alt-text][test_image6]  | -0.225 meters      |

Finally, plot the warped images back down onto the road such that, for each image, the lane area is identified clearly:

![alt text][lane_mapping]

---

### Output Video
 
The video is available on [YouTube](https://youtu.be/yqC3CT6Xx9E) and in the repository as **P4_output_video.mp4**.

### 

---

### Reflections

This project was difficult in that it took a while to organize everything into a functioning system. 
The first few steps required to calibrate the camera, undistort an image and produce a thresholded binary image were straightforward. However, the perspective transform and steps following that proved to be challenging. The video stream was significantly dependent on the color and gradient filters applied to the images, and I feel that if the pipeline were tested on a video taken at night, it would be unable to perform correct lane detections, as the parameters chosen for the color and gradient thresholds are tailored to a video taken during the day.
