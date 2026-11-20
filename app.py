import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model and supporting objects
try:
    model = joblib.load("model/logistic_regression_model.joblib")
    scaler = joblib.load("model/scaler.joblib")
    imputer = joblib.load("model/imputer.joblib")
    feature_columns = joblib.load("model/feature_columns.joblib")
    numeric_cols = joblib.load("model/numeric_cols.joblib")
except Exception as e:
    print(f"Warning: Could not load model files. Error: {e}")
    model = None
    scaler = None
    imputer = None
    feature_columns = []
    numeric_cols = []


def prediksi_nasabah(input_dict, threshold=0.5):
    """
    Predict loan default risk for a single customer.
    input_dict: raw values (not scaled), e.g.:
    {'Age': 25, 'Income': 40000, 'Home': 'RENT', 'Emp_length': 3, 'Intent': 'PERSONAL',
     'Amount': 10000, 'Rate': 13.5, 'Cred_length': 3, 'Prev_Default': 'N'}
    """
    row = pd.DataFrame([input_dict])

    # Percent_income calculated automatically
    row['Percent_income'] = (row['Amount'] / row['Income']).round(2)

    # Encoding same as training
    row['Prev_Default'] = row['Prev_Default'].map({'N': 0, 'Y': 1})
    row = pd.get_dummies(row, columns=['Home', 'Intent'], drop_first=True, dtype=int)

    # Align columns with training data (missing categories filled with 0)
    row = row.reindex(columns=feature_columns, fill_value=0)

    # Impute, scale, then predict
    row[numeric_cols] = imputer.transform(row[numeric_cols])
    row[numeric_cols] = scaler.transform(row[numeric_cols])
    proba = model.predict_proba(row)[0, 1]
    label = 'Berisiko Gagal Bayar' if proba >= threshold else 'Tidak Berisiko'
    return label, proba


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return "Model not loaded. Please train the model using the provided notebook first."

    try:
        input_dict = {
            'Age': int(request.form.get('age', 0)),
            'Income': float(request.form.get('income', 0)),
            'Home': request.form.get('home', 'RENT'),
            'Emp_length': float(request.form.get('emp_length', 0)),
            'Intent': request.form.get('intent', 'PERSONAL'),
            'Amount': float(request.form.get('amount', 0)),
            'Rate': float(request.form.get('rate', 0)),
            'Cred_length': int(request.form.get('cred_length', 0)),
            'Prev_Default': request.form.get('prev_default', 'N'),
        }

        label, proba = prediksi_nasabah(input_dict)
        confidence = round(proba * 100, 2)

        return render_template(
            "result.html",
            prediction=label,
            confidence=confidence,
            is_risky=(label == 'Berisiko Gagal Bayar'),
            input_data=input_dict
        )
    except Exception as e:
        return f"Terjadi kesalahan saat memproses data: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
