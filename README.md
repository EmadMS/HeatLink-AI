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
