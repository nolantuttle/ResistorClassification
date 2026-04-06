#?------------------------------------------------------------
#? Author: Nolan Tuttle & Mathew Hobson
#? Project: ResistorClassification
#?------------------------------------------------------------

import feature_extraction as fe
import train as tr
import predict as pr

prediction = pr.predict_resistor("archive/1K_1-4W/1K_1-4W_(2).jpg")

print(f"Predicted Resistance: {prediction[0][0]}")
print(f"Predicted Wattage: {prediction[0][1]}")