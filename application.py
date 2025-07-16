from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)  # Single app object

# ✅ Lowercase folder path (model/)
scaler = pickle.load(open("model/standardScalar.pkl", "rb"))
model = pickle.load(open("model/modelForPrediction.pkl", "rb"))

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""
    if request.method == 'POST':
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        predict = model.predict(new_data)

        result = 'Diabetic' if predict[0] == 1 else 'Non-Diabetic'
        return render_template('home.html', result=result)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    application.run(host="0.0.0.0")
