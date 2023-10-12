# Import the libraries
import gradio
import joblib
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

# Load your trained model
xgb_model_loaded = joblib.load("xgboost-model.pkl")

def bol_to_int(bol):
  if bol==True:
    return 1
  else:
    return 0

# Function for prediction
def predict_death_event(feature1, feature2, feature3,feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12):
    data = {'age' : [feature1], 'anaemia' : [bol_to_int(feature2)],
            'creatinine_phosphokinase' : [feature3],
            'diabetes' : [bol_to_int(feature4)],
            'ejection_fraction' : [feature5],
            'high_blood_pressure' : [bol_to_int(feature6)],
            'platelets' : [feature7],
            'serum_creatinine' : [feature8],
            'serum_sodium' : [feature9],
            'sex' : [bol_to_int(feature10)],
            'smoking' : [bol_to_int(feature11)],
            'time' : [feature12]}
    df = pd.DataFrame(data)
    y_pred = xgb_model_loaded.predict(df)[0]
    return y_pred

# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs=[
                            gradio.components.Slider(30, 100, step=1, label= 'age'),
                            gradio.components.Radio(["0","1"], label= 'anaemia'),
                            gradio.components.Slider(1, 10000, step=1, label= 'creatinine_phosphokinase'),
                            gradio.components.Radio(["0","1"], label= 'diabetes'),
                            gradio.components.Slider(1, 100, step=1, label= 'ejection_fraction'),
                            gradio.components.Radio(["0","1"], label= 'high_blood_pressure'),
                            gradio.components.Number(label= 'platelets'),
                            gradio.components.Slider(0.1, 10.0, step=0.1, label= 'serum_creatinine'),
                            gradio.components.Slider(100, 150, step=1, label= 'serum_sodium'),
                            gradio.components.Radio(["0","1"], label= 'sex'),
                            gradio.components.Radio(["0","1"], label= 'smoking'),
                            gradio.components.Slider(1, 300, step=1, label= 'time')],
                         outputs = [gradio.components.Textbox (label ='DeathEvent')],
                         title = title,
                         description = description)

#iface.launch(debug=True, share = True)
iface.launch(server_name = "0.0.0.0", server_port = 8001) # Ref. for parameters: https://www.gradio.app/docs/interface