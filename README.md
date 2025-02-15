# HeatLink-AI

## Installation
  - Docker: [danirali2007/heatlink-ai](https://hub.docker.com/r/danirali2007/heatlink-ai)
  - Current Tag: development-v0.0.1

## Usage

 ### Heating Test Data
   ##### Uploading heating_test_data.json
    curl -X POST -F "file=@heating_test_data.json" http://127.0.0.1:8080/upload_heating_data
 ### Using AI Model
   ##### Prediction Request
    curl -X POST -H "Content-Type: application/json" -d '{"ip_address": "10.0.0.99"}' http://127.0.0.1:8080/predict_boiler  
## Development

  ### Useful Files
    train_model.py - Train AI Model with random data
    gen_dummy-thermostat-data.py - Generate Dummy Thermostat Data
    instance/boiler_model.pth - AI Model
  
  ### Heating Test Data File
    {
        {
        "hourly": {
            "time": ["2025-02-15T12:00", "2025-02-15T13:00", "2025-02-15T14:00", "2025-02-15T15:00"],
            "temperature_2m": [5.0, 6.0, 7.5, 8.0],
            "apparent_temperature": [2.0, 3.0, 4.5, 5.0],
            "precipitation": [0.1, 0.05, 0.0, 0.0]
        }
    }
