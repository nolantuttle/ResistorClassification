#?------------------------------------------------------------
#? Author: Nolan Tuttle & Mathew Hobson
#? Project: ResistorClassification
#?------------------------------------------------------------

import feature_extraction as fe
import train as tr
import predict as pr

fe.extract_features('archive')
tr.train_model()
prediction = pr.predict_resistor("test_image.jpg")

print(f"Predicted Resistance: {prediction[0][0]}")
print(f"Predicted Wattage: {prediction[0][1]}")