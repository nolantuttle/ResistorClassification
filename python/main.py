#?------------------------------------------------------------
#? Author: Nolan Tuttle & Mathew Hobson
#? Project: ResistorClassification
#?------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import cv2 as cv
import numpy as np

x = list()
y_resistance = list()
y_wattage = list()

# os.walk traverse through the dataset folder, join each file name and filepath into a full filename for loading
for root, dirs, files in os.walk('../archive'):
    for file in files:
        if (file.endswith('.jpg')):
            resistance = (file.split("_")[0])
            wattage = (file.split("_")[1])
            image = cv.imread(os.path.join(root, file))
            image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            image = cv.resize(image, (700, 700))
            hsv_planes = cv.split(image)

            histSize = 180;
            histRange = (0, 180)
            hue_hist = cv.calcHist(hsv_planes, [0], None, [histSize], histRange, False)
            hue_hist = hue_hist.flatten()

            histSize = 255
            histRange = (0, 256)
            sat_hist = cv.calcHist(hsv_planes, [1], None, [histSize], histRange, False)
            sat_hist = sat_hist.flatten()

            x_hist = np.concatenate((hue_hist, sat_hist))

            y_resistance.append(resistance)
            y_wattage.append(wattage)
            x.append(x_hist)

x = np.array(x)
y_resistance = np.array(y_resistance)
y_wattage = np.array(y_wattage)

x_train, x_test, y_resistance_train, y_resistance_test, y_wattage_train, y_wattage_test = train_test_split(x, y_resistance, y_wattage, test_size=0.2, random_state=42)
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

y_train_scaled = np.column_stack((y_resistance_train, y_wattage_train))

model = MultiOutputClassifier(SVC())
model.fit(x_train_scaled, y_train_scaled)

with open("model_and_scaler.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

y_pred = model.predict(x_test_scaled)

accuracy1 = accuracy_score(y_resistance_test, y_pred[:, 0])
print(f"\nAccuracy 1: {accuracy1 * 100:.2f}%")
accuracy2 = accuracy_score(y_wattage_test, y_pred[:, 1])
print(f"\nAccuracy 2: {accuracy2 * 100:.2f}%")

print("\nClassification Report for Resistance:")
print(classification_report(y_resistance_test, y_pred[:, 0], zero_division=0))
print("\nClassification Report for Wattage:")
print(classification_report(y_wattage_test, y_pred[:, 1], zero_division=0))

