# 🔥 **HeatLink AI Model**  

**HeatLink AI** is an advanced machine learning model designed to predict **boiler run time, temperature, and energy usage** based on real-time environmental data. It leverages weather conditions, room temperature, and time-based patterns to optimize heating efficiency.  

---

## **🚀 Features**  
✅ **Predicts Boiler Run Time** – Ensures heating operates within a set range (50-200 mins).  
✅ **Temperature Control** – Keeps boiler temperature within safe limits (0-40°C).  
✅ **Real-Time Weather Integration** – Uses external APIs to fetch hourly temperature, precipitation, and apparent temperature.  
✅ **Flask API** – Deployable REST API for seamless integration with smart home systems.  
✅ **SQL Database Support** – Stores thermostat data for better control and management.  

---

## **🔧 Technologies Used**  
- **Python (Flask, SQLAlchemy, NumPy, Torch)**  
- **Machine Learning (PyTorch, Neural Networks)**  
- **Weather Data API (Open-Meteo)**  
- **SQLite Database**  

---

## **⚡ How It Works**  
1️⃣ The model collects **real-time weather data** (temperature, precipitation, time of day).  
2️⃣ It processes room temperature and **target temperature** as input features.  
3️⃣ The **AI model predicts** the necessary boiler run time and temperature.  
4️⃣ Outputs are **clamped to realistic ranges** (boiler temp: **0-40°C**, runtime: **50-200 min**).  
5️⃣ The Flask API serves predictions for **integration with home automation systems**.  

---

## **🛠️ Installation**  
```bash
git clone https://github.com/EmadMS/heatlink-ai.git  
cd heatlink-ai  
pip install -r requirements.txt  
python main.py  
```

---

## **📡 API Endpoints**  
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict_boiler` | **POST** | Predicts boiler runtime & temperature |
| `/register_thermostat` | **POST** | Registers a new thermostat device |
| `/upload_heating_data` | **POST** | Uploads test heating data |

---

## **📌 Future Enhancements**  
🔹 Adaptive learning for improved accuracy  
🔹 Support for multiple heating zones  
🔹 Energy efficiency optimizations  

---

💡 **Contributions & feedback are welcome!** Let's build smarter heating solutions together. 🚀🔥  

👉 **[GitHub Repository](https://github.com/EmadMS/heatlink-ai)** 
