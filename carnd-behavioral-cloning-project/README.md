# Behavioral Cloning - Teaching a Car to Learn Driving Patterns through Deep Learning

Application of Deep Learning to the task of autonomous vehicle driving. Udacity provided a training simulator to gather the training data and a simulator to test the model predictions on the same track.

The model was trained on only Track 1. It performs well on Track 1 and completed the entire laps on it. Surprisingly, the model also covered a major portion os the Track 2 without having driven on it for even a single time. This proves that the model has generalized well on the training data.

## Data Collection

I bought an analog controller online after looking at a lot of suggestions on the course forum and Slack channels and used the 10hz simulator to collect around 40,000 examples in my first attempt. However, my training data had a lot of bias. This was because most of the data had either zero sterring angles or a negative value as majority of the track involved turning left.

I tried collecting more data with smoother steering and tried to avoid making very sharp turns and instead focused on making small and smoother turns with a moderate steering angle. This of course worked for a majority of the track and sharp turns were reduced to only when necessary.

The next step I took was to drive the car in the opposite direction as well for an equal number of laps. This corrected the left steering bias in the data and provided me with a well balanced dataset.

I also added the "rescuing" techniques after some suggestions on the course forums, which means I drove close to the edge and then directed the car towards the middle of the lane.

Overall in my second attempt I drove around the track for 20 laps (around 45-50 minutes), with 7-8 laps in the forwards direction, 7-8 laps in the opposite direction and 5-6 laps of going near the lanes and then steering back. Finally I ended up with more than 72,500 images for training.

## Data Preprocessing

I cropped the images along the y-axis to remove the parts above the horizon and the parts of the car's bonnet. I made no changes along the x-axis.

I also resized the images to (66, 200) to match nvidia's paper and normalize it.

My input images therefore now look like this:
![processed input image](figure_1.png)

I have used the data augmentation technique, described by Vivek Yadav, another student, to generate additional data. I'm using left, right and center images randomly with a steering shift value of 0.2 and flipping the images in 50% of the cases.

To take care of the high number of steering angles being close to 0, I removed around 60 percent of the values in the interval of [-0.02, 0.02].

I finally split my dataset into train/validation set with a factor of 0.75 (which means 75% is training, 25% is validation/test).

## Network and Training

I made use of the generous grant of 50 USD from Udacity for AWS and trained using a GPU enabled machine.

The data is fed into a Keras `fit_generator`. The generator picks a random batch of the dataset, picks one of the images (left, center, right) and flips the image in some cases. Then the generator yields the batch and feeds it to the network. This continues until the whole dataset is used.

I decided to use a model based on [Nvidia's paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). To reduce overfitting, several dropout layers have been added to the network.

![Network](model-visualization.png)

The network consists of five convolutional layers, followed by three fully connected layers. I have added Dropout Layers and SpatialDropout Layers to prevent overfitting. The `model.summary()` command prints the following output:

```
____________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
lambda_2 (Lambda)                (None, 66, 200, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 100, 24)   1824        lambda_2[0][0]
____________________________________________________________________________________________________
spatialdropout2d_1 (SpatialDropo (None, 33, 100, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 50, 36)    21636       spatialdropout2d_1[0][0]
____________________________________________________________________________________________________
spatialdropout2d_2 (SpatialDropo (None, 17, 50, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 23, 48)     43248       spatialdropout2d_2[0][0]
____________________________________________________________________________________________________
spatialdropout2d_3 (SpatialDropo (None, 7, 23, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 21, 64)     27712       spatialdropout2d_3[0][0]
____________________________________________________________________________________________________
spatialdropout2d_4 (SpatialDropo (None, 5, 21, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 19, 64)     36928       spatialdropout2d_4[0][0]
____________________________________________________________________________________________________
spatialdropout2d_5 (SpatialDropo (None, 3, 19, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 3648)          0           spatialdropout2d_5[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 3648)          0           flatten_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           364900      dropout_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_2[0][0]
====================================================================================================
Total params: 501,819
Trainable params: 501,819
Non-trainable params: 0
____________________________________________________________________________________________________

```

I have used lambda layers to normalize the data and resize it. This avoids the need to add these things in *drive.py* because they are included in the network itself.

I'm using an Adam optimizer with a learning rate 0.001.

I tried training with different epochs from 25 to 100. I found that the loss on the training and validation sets is mostly constant after 60 epochs. I have therefore trained my dataset for 75 epochs with a batch size of 100.

## Testing and Autonomous Driving

In the first few attempts with my old dataset, the car would dangerously oscillate about the track for a few seconds and then finally veer off the track. This was mainly due to left and zero biases in the data.

After training my model in the new dataset, and adding the image trimming, flipping and translations functions, the car was able to complete a few laps in the first track. However, I was not satisifed as it was still making a lot of jittery turns and was taking very sharp turns near the mud tracks. It was also driving dangerously close to the lane edges for many portions of the track.

Therefore, I made a few changes in my hyper-parameters like reduceing the batch size, increasing the number of epochs and incorporating more data with low steering angles (to reduce the jitter).

Finally, the car was able to smoothly travel around the track 1 for many laps and was able to generalize around a major part of the track 2 as well.

## Tools Setup

The simulator is available at [udacity's github page](https://github.com/udacity/self-driving-car-sim).

### Important Note:
When I did this project, the simulator required two files: a json file and a h5 file with the model weights. So the code saves the files differently.
Now the simulator requires only a single json file with all the weights. I will make the changes in the code to reflect this soon.

