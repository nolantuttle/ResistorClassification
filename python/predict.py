import pickle
import cv2 as cv
import numpy as np
import feature_extraction as fe

with open("model_and_scaler.pkl", "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        scaler = data["scaler"]

def predict_resistor(image_path):
    x_hist = fe.extract_feature_single_image(image_path)
    x_hist = np.array(x_hist)
    x_hist_scaled = scaler.transform([x_hist])
    prediction = model.predict(x_hist_scaled)

    return prediction