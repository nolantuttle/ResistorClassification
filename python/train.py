import feature_extraction as fe
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
import pickle
import numpy as np

def train_model():

    x, y_resistance, y_wattage = fe.extract_features('archive')

    x_train, x_test, y_resistance_train, y_resistance_test, y_wattage_train, y_wattage_test = train_test_split(x, y_resistance, y_wattage, test_size=0.2, random_state=42)
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_train_scaled = np.column_stack((y_resistance_train, y_wattage_train))

    model = MultiOutputClassifier(SVC(kernel='linear', probability=True))
    model.fit(x_train_scaled, y_train_scaled)

    with open("model_and_scaler.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    
    return model, scaler