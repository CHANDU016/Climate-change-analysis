from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', output="⚠️ Please upload a CSV file.")

    try:
        df = pd.read_csv(file)

        # Optional: Drop 'Unnamed: 0' if it appears
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        df = df.dropna()

        features_to_drop = [
            'SO2 solar zenith angle', 'CO solar zenith angle', 'NO2 solar zenith angle',
            'Formaldehyde solar zenith angle', 'Ozone solar zenith angle', 'Cloud solar zenith angle',
            'SO2 sensor azimuth angle', 'CO sensor azimuth angle', 'NO2 sensor azimuth angle',
            'Formaldehyde sensor azimuth angle', 'Ozone sensor azimuth angle', 'Cloud sensor azimuth angle',
            'ID LAT LON YEAR WEEK', 'year', 'week no'
        ]

        # Drop only existing columns
        df = df.drop(columns=[col for col in features_to_drop if col in df.columns])

        if 'emission' not in df.columns:
            return render_template('index.html', output="❌ 'emission' column missing.")

        X = df.drop('emission', axis=1)

        y_pred = model.predict(X)

        df['Prediction'] = y_pred

        df['error'] = y_pred - df['emission']

        result_html = df.to_html(classes='data', index=False)


        return render_template('index.html', output=result_html)

    except Exception as e:
        return render_template('index.html', output=f"❌ Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
