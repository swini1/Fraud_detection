from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("fraud_model.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    amount = float(request.form['amount'])
    oldbalance = float(request.form['oldbalance'])
    newbalance = float(request.form['newbalance'])

    features = np.array([[amount,oldbalance,newbalance]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Fraud Transaction Detected"
    else:
        result = "Transaction is Safe"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)