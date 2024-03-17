from flask import Flask, request, render_template
import pandas as pd
import os
import joblib
import numpy as np
import math

app = Flask(__name__)

# Load the trained Tweedie regression model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'tweedie_regression_model.pkl')
optimized_model = joblib.load(model_path)

# Load the processed data
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
csv_file = os.path.join(static_folder, 'processed_data.csv')
processed_data = pd.read_csv(csv_file)
csv_lots_file = os.path.join(static_folder, 'my_data.csv')
car_data = pd.read_csv(csv_lots_file)
car_parks = processed_data['car_park_no'].unique()

# Columns to use as features
selected_columns = ['update_datetime', 'car_park_type','free_parking','night_parking','car_park_decks','gantry_height']  # Replace with the actual column names you want to use

# Calculate the min and max values of update_datetime
min_datetime = pd.to_datetime(processed_data['update_datetime'].min())
max_datetime = pd.to_datetime(processed_data['update_datetime'].max())

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        car_park_number = request.form['car_park']
        date = pd.to_datetime(request.form['date'])  # Convert date to Pandas Timestamp object
        
        # Calculate normalized update_datetime value
        normalized_date = normalize_date(date, min_datetime, max_datetime)
        
        # Update the record with normalized update_datetime
        record = processed_data[processed_data['car_park_no'] == car_park_number].iloc[0]
        record['update_datetime'] = normalized_date
        
        # Extract features from the record for prediction
        features = record[selected_columns]
        
        # Make predictions using the loaded model
        prediction = optimized_model.predict(features.values.reshape(1, -1))  # Extract the single prediction value
        
        return render_template('submitted.html', car_park_number=car_park_number, date=date, prediction=int(round(prediction[0] * car_data[car_data['car_park_no'] == car_park_number]['total_lots'].values[0],0)),
                                total_lots=car_data[car_data['car_park_no'] == car_park_number]['total_lots'].values[0])
    
    return render_template('form.html', car_parks=car_parks)

def normalize_date(date, min_date, max_date):
    # Convert date to timestamp
    date_timestamp = date.timestamp()
    min_timestamp = min_date.timestamp()
    max_timestamp = max_date.timestamp()
    
    # Normalize the date
    if max_timestamp == min_timestamp:
        normalized_date = 0
    else:
        normalized_date = (date_timestamp - min_timestamp) / (max_timestamp - min_timestamp)
    
    return normalized_date

if __name__ == '__main__':
    app.run(debug=True)
