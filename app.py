from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model  = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['bloodpressure']),
        float(request.form['skinthickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]
    data = scaler.transform([features])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    result = 'Diabetic' if prediction == 1 else 'Not Diabetic'
    confidence = round(probability * 100, 2)
    return render_template('result.html',
                           result=result,
                           confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)