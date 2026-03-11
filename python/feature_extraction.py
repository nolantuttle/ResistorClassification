import os
import cv2 as cv
import numpy as np
import scipy

def extract_features(filepath):
    x = list()
    y_resistance = list()
    y_wattage = list()

    for root, dirs, files in os.walk(filepath):
        for file in files:
            if (file.endswith('.jpg')):
                resistance = (file.split("_")[0])
                wattage = (file.split("_")[1])
                image = cv.imread(os.path.join(root, file))
                image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                image = cv.resize(image, (700, 700))
                hsv_planes = cv.split(image)

                # here we are appending the mean, std, and skew of the 2d arrays of pixel values to show distribution of pixel values
                hsv_mean = [np.mean(hsv_planes[0]), np.mean(hsv_planes[1]), np.mean(hsv_planes[2])]
                hsv_std = [np.std(hsv_planes[0]), np.std(hsv_planes[1]), np.std(hsv_planes[2])]
                hsv_skew = [scipy.stats.skew(hsv_planes[0].flatten()), scipy.stats.skew(hsv_planes[1].flatten()), scipy.stats.skew(hsv_planes[2].flatten())]

                # hue histogram
                histSize = 180;
                histRange = (0, 180)    # hue values range from 0 to 179 in OpenCV
                hue_hist = cv.calcHist(hsv_planes, [0], None, [histSize], histRange, False)
                hue_hist = hue_hist.flatten()

                # saturation histogram
                histSize = 255
                histRange = (0, 256)   # saturation and brightness values range from 0 to 255 in OpenCV
                sat_hist = cv.calcHist(hsv_planes, [1], None, [histSize], histRange, False)
                sat_hist = sat_hist.flatten()

                # brightness histogram
                histSize = 255
                histRange = (0, 256)
                bright_hist = cv.calcHist(hsv_planes, [2], None, [histSize], histRange, False)
                bright_hist = bright_hist.flatten()

                # concatenate all the features into a single feature vector for the model
                x_hist = np.concatenate((hue_hist, sat_hist, bright_hist, hsv_mean, hsv_std, hsv_skew))

                # append the feature vector and labels to the respective lists
                y_resistance.append(resistance)
                y_wattage.append(wattage)
                x.append(x_hist)

    # convert lists to numpy arrays for model training
    x = np.array(x)
    y_resistance = np.array(y_resistance)
    y_wattage = np.array(y_wattage)

    return x, y_resistance, y_wattage
    
def extract_feature_single_image(image_path):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image = cv.resize(image, (700, 700))
    hsv_planes = cv.split(image)

    hsv_mean = [np.mean(hsv_planes[0]), np.mean(hsv_planes[1]), np.mean(hsv_planes[2])]
    hsv_std = [np.std(hsv_planes[0]), np.std(hsv_planes[1]), np.std(hsv_planes[2])]
    hsv_skew = [scipy.stats.skew(hsv_planes[0].flatten()), scipy.stats.skew(hsv_planes[1].flatten()), scipy.stats.skew(hsv_planes[2].flatten())]

    histSize = 180;
    histRange = (0, 180)    # hue values range from 0 to 179 in OpenCV
    hue_hist = cv.calcHist(hsv_planes, [0], None, [histSize], histRange, False)
    hue_hist = hue_hist.flatten()

    histSize = 255
    histRange = (0, 256)   # saturation and brightness values range from 0 to 255 in OpenCV
    sat_hist = cv.calcHist(hsv_planes, [1], None, [histSize], histRange, False)
    sat_hist = sat_hist.flatten()

    histSize = 255
    histRange = (0, 256)
    bright_hist = cv.calcHist(hsv_planes, [2], None, [histSize], histRange, False)
    bright_hist = bright_hist.flatten()

    x_hist = np.concatenate((hue_hist, sat_hist, bright_hist, hsv_mean, hsv_std, hsv_skew))

    return x_hist