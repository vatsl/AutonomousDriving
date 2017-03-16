### Importing packages.
import os
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2
import math
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers import Input, ELU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras import initializations

from pathlib import Path
import json
%matplotlib inline

base_dir = os.getcwd()
def load_driving_logs(log_file_path):
    """
    Load the log file and the images
    """
    x_var = []
    y_var = []
    logs = []
    with open(log_file_path, 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            logs.append(line)
            imgc_file = log_file_path + line[0].strip()
            imgl_file = log_file_path + line[1].strip()
            imgr_file = log_file_path + line[2].strip()
            # checking for the image files and adding small steering angles for off_center images
            if (os.path.isfile(imgc_file) & os.path.isfile(imgl_file) & os.path.isfile(imgr_file)):
                x_var.append(imgc_file)
                y_var.append(np.float32(line[3]))
                x_var.append(imgl_file)
                y_var.append(np.float32(line[3]) + 0.2)
                x_var.append(imgr_file)
                y_var.append(np.float32(line[3]) - 0.2)
    log_labels = logs.pop(0)
    print(len(logs))
    print(len(y_var))
    return x_var, y_var

def load_image(f):
    """
    Load image and convert it to RGB format since cv2.imread() reads in a BGR image
    :param f: file path
    :return: RGB image
    """
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def augment_brightness(image):
    """
    Simulate various brightness conditions
    :return: gamma_corrected image
    """
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    return image1

def trans_image(image, steer, trans_range):
    """
    Translation - shift camera images to left and right to simulate the effect of the car
    at different positions in the lane. To simulate lane shifts, apply random shifts
    in horizontal direction of upto 10 pixels, and apply angle change of .2 per pixel.
    """
    rows,cols,channels = image.shape
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 10 * np.random.uniform() - 10 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang, tr_x

def preprocessImage(image):
    """
    Crop images and reduce their dimensions
    :param image: raw image
    :return: processed image
    """
    shape = image.shape
    new_rows = 64
    new_cols = 64
    # note: numpy arrays are (row, col)!
    h = np.floor((shape[0]/3)-5)
    w = shape[0]-25
    img_trimed = image[int(h):int(w), 0:int(shape[1])]
    img_resized = cv2.resize(img_trimed, (new_rows, new_cols), interpolation=cv2.INTER_AREA)
    return img_resized

def generate_image(image, y):
    """
    Perform the cropping, resizing, brightness augmentation, translation and flipping functions on the image:
    """
    steer_angle = y

    img_processed = preprocessImage(image)
    image_gamma_corrected = augment_brightness(img_processed)
    image_final, y_steer, tr_x = trans_image(image_gamma_corrected, steer_angle, 150)

    if (np.random.uniform() > 1.0):
        image_final = cv2.flip(image_final, 1)
        y_steer= -1.0 * y_steer

    return (image_final, y_steer)

def preprocess_image_file_train(line_data):
    # Preprocessing training files and augmenting
    i_lrc = np.random.randint(3)
    if (i_lrc == 0):
        path_file = line_data['left'][0].strip()
        shift_ang = .2
    if (i_lrc == 1):
        path_file = line_data['center'][0].strip()
        shift_ang = 0.
    if (i_lrc == 2):
        path_file = line_data['right'][0].strip()
        shift_ang = -.2

    y_steer = line_data['steer'][0] + shift_ang
    image = load_image(path_file)
    final_image, final_y = generate_image(image, y_steer)

    return final_image, final_y

def preprocess_image_file_predict(line_data):
    # Preprocessing Prediction files and augmenting
    path_file = line_data['center'][0].strip()
    #print(path_file)
    image = load_image(path_file)
    #image = cv2.imread(path_file)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = preprocessImage(image)
    image = np.array(image)

    return image

pr_threshold = 1
def generate_train_from_PD_batch(data, batch_size=32):
    ## Generator for keras training, with subsampling
    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[[i_line]].reset_index()

            keep_pr = 0
            # x,y = preprocess_image_file_train(line_data)
            while keep_pr == 0:
                x, y = preprocess_image_file_train(line_data)
                pr_unif = np.random
                if abs(y) < .05:
                    pr_val = np.random.uniform()
                    if pr_val > pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1

            # x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            # y = np.array([[y]])
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering

def generate_train_from_PD(data):
    # Old generator, not used
    while 1:
        i_line = np.random.randint(len(data))
        line_data = data.iloc[[i_line]].reset_index()
        x, y = preprocess_image_file_train(line_data)
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        y = np.array([[y]])
        yield x, y

def generate_valid_from_PD(data):
    # Validation generator
    while 1:
        for i_line in range(len(data)):
            line_data = data.iloc[[i_line]].reset_index()
            # print(line_data)
            x = preprocess_image_file_predict(data)
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            y = line_data['steer'][0]
            y = np.array([[y]])
            yield x, y

def get_model():
    input_shape = (new_size_row, new_size_col, 3)
    filter_size = 3
    pool_size = (2,2)
    model = Sequential()
    model.add(Lambda(lambda x: x/255.-0.5,input_shape=input_shape))
    model.add(Convolution2D(3,1,1, border_mode='valid', name='conv0', init='he_normal'))

    model.add(Convolution2D(32,filter_size,filter_size, border_mode='valid', name='conv1', init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(32,filter_size,filter_size, border_mode='valid', name='conv2', init='he_normal'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64,filter_size,filter_size, border_mode='valid', name='conv3', init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(64,filter_size,filter_size, border_mode='valid', name='conv4', init='he_normal'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128,filter_size,filter_size, border_mode='valid', name='conv5', init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(128,filter_size,filter_size, border_mode='valid', name='conv6', init='he_normal'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512,name='hidden1', init='he_normal'))
    model.add(ELU())

    model.add(Dropout(0.5))
    model.add(Dense(64,name='hidden2', init='he_normal'))
    model.add(ELU())

    model.add(Dropout(0.5))
    model.add(Dense(16,name='hidden3',init='he_normal'))
    model.add(ELU())

    model.add(Dropout(0.5))
    model.add(Dense(1, name='output', init='he_normal'))
    return model

def save_model(fileModelJSON,fileWeights):
    #print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON,'w' ) as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)

## Defining variables
pr_threshold = 1
new_size_col = 64
new_size_row = 64

base_dir = os.getcwd()
csv_path = os.path.join(base_dir, 'driving_log.csv')
print(csv_path)

data_files_s = pd.read_csv(csv_path, index_col = False)
data_files_s.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']

rev_steer_s = np.array(data_files_s.steer,dtype=np.float32)

t_s = np.arange(len(rev_steer_s))
x_s = np.array(data_files_s.steer)
y_s = rev_steer_s

steer_sm_s = rev_steer_s
print(len(rev_steer_s))

# Define model

model = get_model()
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')

valid_s_generator = generate_valid_from_PD(data_files_s)

## Training loop,
## Saving all intermediate files.
### pr_threshold reduced over training to include more small angles
val_size = len(data_files_s)
print(val_size)
pr_threshold = 1

batch_size = 256

i_best = 0
val_best = 1000

for i_pr in range(6,16):

    train_r_generator = generate_train_from_PD_batch(data_files_s, batch_size)

    #nb_vals = np.round(len(data_files_s) / val_size) - 1
    #print(nb_vals)
    history = model.fit_generator(train_r_generator,samples_per_epoch=44720,
                                  nb_epoch=3, validation_data=valid_s_generator,
                                  nb_val_samples=val_size)

    fileModelJSON = 'model_' + str(i_pr) + '.json'
    fileWeights = 'model_' + str(i_pr) + '.h5'

    save_model(fileModelJSON, fileWeights)

    val_loss = history.history['val_loss'][0]
    if val_loss < val_best:
        i_best = i_pr
        val_best = val_loss
        fileModelJSON = 'model_best.json'
        fileWeights = 'model_best.h5'
        save_model(fileModelJSON, fileWeights)

    pr_threshold = 1 / (i_pr + 1)
print('Best model found at iteration # ' + str(i_best))
print('Best Validation score : ' + str(np.round(val_best, 4)))
