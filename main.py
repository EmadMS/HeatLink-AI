from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import requests
import json
import threading
import time
import torch
import torch.nn as nn
import numpy as np
import os
from datetime import datetime
# Configuration
PORT = 8080
POLLING_RATE = 3  # Fetch weather data every 3 hours
TEST_DATA_FILE = "instance/weather_test_data.json"

web_ui = Flask(__name__)
web_ui.secret_key = 'your_secret_key_here'

web_ui.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
web_ui.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(web_ui)

# Weather API
API_URL = "https://api.open-meteo.com/v1/forecast?latitude=51.515673855942595&longitude=-0.4120755176963609&hourly=temperature_2m,apparent_temperature,precipitation,rain&wind_speed_unit=mph&timezone=auto&forecast_days=14"

# AI Model Definition (Heating Predictor)
class HeatingPredictor(nn.Module):
    def __init__(self):
        super(HeatingPredictor, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)  # Output: [Run Time (mins), Boiler Temperature]
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Load AI model once when the app starts

model = HeatingPredictor()
try:
    if os.path.exists("instance/boiler_model.pth"):
        model.load_state_dict(torch.load("instance/boiler_model.pth"))
        model.eval()
        print("AI Model Loaded Successfully.")
    else:
        print("Model file not found. Please train the model first.")
except Exception as e:
    print(f"Error loading AI model: {e}")

# Database model for thermostats
class Thermostat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(100), unique=True, nullable=False)

# Fetch weather data (Background Task)
def fetch_weather_data():
    while True:
        try:
            response = requests.get(API_URL)
            response.raise_for_status()
            data = response.json()

            # Save weather data to a file
            os.makedirs("instance", exist_ok=True)  # Ensure the directory exists
            with open("instance/weather_data.json", "w") as file:
                json.dump(data, file, indent=4)

            print("Weather data updated.")

        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            # Optionally: Save a fallback empty JSON or return an error message
            with open("instance/weather_data.json", "w") as file:
                json.dump({}, file)

        time.sleep(POLLING_RATE * 3600)  # Wait for next fetch

# Function to simulate fetching room temperature
def get_room_temperature(ip_address):
    return np.random.uniform(15, 25)  # Simulated thermostat reading

# API to register a thermostat
@web_ui.route('/register_thermostat', methods=['POST'])
def register_thermostat():
    data = request.json
    ip_address = data.get('ip_address')

    if not ip_address:
        return jsonify({'error': 'IP Address is required'}), 400

    if not Thermostat.query.filter_by(ip_address=ip_address).first():
        db.session.add(Thermostat(ip_address=ip_address))
        db.session.commit()

    return jsonify({'message': 'Thermostat registered', 'ip_address': ip_address})

# API to upload test heating data
@web_ui.route('/upload_heating_data', methods=['POST'])
def upload_heating_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Validate file type (ensure it's a JSON file)
    if not file.filename.endswith('.json'):
        return jsonify({'error': 'File must be a JSON file'}), 400

    # Save the file
    try:
        os.makedirs("instance", exist_ok=True)  # Ensure the directory exists
        file.save(TEST_DATA_FILE)
        return jsonify({'message': 'Test heating data uploaded successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {e}'}), 500

# API to predict heating power
@web_ui.route('/predict_boiler', methods=['POST'])
def predict_boiler():
    data = request.json
    ip_address = data.get('ip_address')
    target_temp = data.get('target_temp', 21.5)

    if not ip_address:
        return jsonify({'error': 'IP Address is required'}), 400

    room_temp = get_room_temperature(ip_address)

    # Get weather data
    with open("instance/weather_data.json", "r") as file:
        weather_data = json.load(file)

    # Get the current hour's index
    current_hour = datetime.utcnow().hour  # UTC hour index

    outside_temp = weather_data["hourly"]["temperature_2m"][current_hour]
    apparent_temp = weather_data["hourly"]["apparent_temperature"][current_hour]
    future_temp_change = weather_data["hourly"]["temperature_2m"][current_hour + 3] - outside_temp  # Change in 3 hrs
    precipitation = weather_data["hourly"]["precipitation"][current_hour]

    # Convert time to sine and cosine encoding
    time_sin = np.sin(2 * np.pi * current_hour / 24)
    time_cos = np.cos(2 * np.pi * current_hour / 24)


    # Normalize inputs (Example: Min-Max Scaling)
    min_temp, max_temp = 0, 40  # Example min/max values for temperatures (modify according to your dataset)
    normalized_room_temp = (room_temp - min_temp) / (max_temp - min_temp)
    normalized_outside_temp = (outside_temp - min_temp) / (max_temp - min_temp)
    normalized_apparent_temp = (apparent_temp - min_temp) / (max_temp - min_temp)

    # Prepare input with normalized features
    input_data = torch.tensor([[normalized_room_temp, target_temp, normalized_outside_temp, normalized_apparent_temp, future_temp_change, precipitation, time_sin, time_cos]], dtype=torch.float32)

    # Get prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(input_data)  # Model output is a tensor of shape (1, 2)
    
    predicted_run_time = max(output[0][0].item(), 0)  # Prevent negative runtime
    predicted_boiler_temp = max(output[0][1].item(), 0)  # Prevent negative temperature if needed


    # Denormalize output if necessary (if the model outputs are normalized)
    predicted_run_time = predicted_run_time * 60  # Example: If run time was scaled to hours, scale it back to minutes
    predicted_boiler_temp = predicted_boiler_temp * (max_temp - min_temp) + min_temp  # Example: Denormalize boiler temp
    # Estimate predicted energy usage (Example formula: run time * boiler temperature)
    predicted_energy_usage = (predicted_run_time * predicted_boiler_temp) / 1000  # Adjust based on your system's energy formula

    return jsonify({
        'room_temperature': round(room_temp, 2),
        'outside_temperature': outside_temp,
        'predicted_run_time_minutes': round(predicted_run_time, 3),
        'predicted_boiler_temperature': round(predicted_boiler_temp, 1),
        'predicted_energy_usage': round(predicted_energy_usage, 2)  # Returns energy usage
    })




# Home Page
@web_ui.route('/')
def main():
    return render_template('index.html')

# Start background thread for weather fetching
threading.Thread(target=fetch_weather_data, daemon=True).start()

if __name__ == '__main__':
    with web_ui.app_context():
        db.create_all()
    web_ui.run(debug=True, port=PORT, host="0.0.0.0")
