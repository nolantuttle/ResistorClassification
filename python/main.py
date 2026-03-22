#?------------------------------------------------------------
#? Author: Nolan Tuttle & Mathew Hobson
#? Project: ResistorClassification
#?------------------------------------------------------------

import feature_extraction as fe
import train as tr
import predict as pr

tr.train_model('archive_clean')
prediction = pr.predict_resistor("470.jpg")

print(f"Predicted Resistance: {prediction[0][0]}")
print(f"Predicted Wattage: {prediction[0][1]}")