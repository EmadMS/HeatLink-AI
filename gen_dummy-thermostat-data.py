import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime

# Dummy Data Function
def generate_dummy_data(num_samples=1000):
    np.random.seed(42)
    timestamps = [datetime(2024, 1, 1, hour=i % 24) for i in range(num_samples)]
    
    # Simulate room and outside temperatures
    room_temps = np.random.uniform(15, 25, num_samples)  # Simulated thermostat readings
    outside_temps = np.random.uniform(-5, 15, num_samples)  # Simulated weather data
    apparent_temps = outside_temps - np.random.uniform(0, 3, num_samples)
    
    # Precipitation and time of day
    precipitation = np.random.uniform(0, 5, num_samples)  # Rain/snow in mm
    time_of_day = np.array([t.hour for t in timestamps])
    
    # Convert time of day to sin/cos representation (cyclic feature)
    time_sin = np.sin(2 * np.pi * time_of_day / 24)
    time_cos = np.cos(2 * np.pi * time_of_day / 24)
    
    # Simulate future temperature change based on external temperature trends
    future_temp_change = np.random.uniform(-2, 2, num_samples)  # Temperature change in the next few hours
    
    # Target temperature (e.g., 21.5°C)
    target_temp = np.full(num_samples, 21.5)

    # Heating Power Calculation (Run Time and Temperature Setpoint)
    # Adjust run time based on temperature difference and future temperature changes
    heating_power = np.clip(20 - room_temps, 0, 15) + np.random.normal(0, 1, num_samples)
    
    # Simulate predicted heating run time based on external temperature trends and target
    run_time = np.clip((target_temp - room_temps) * 10 + np.random.uniform(10, 30, num_samples), 10, 60)
    
    # Boiler temperature: setpoint should be based on target temperature and outside temperature
    boiler_temp = target_temp - np.random.uniform(0, 1, num_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        "room_temp": room_temps,
        "outside_temp": outside_temps,
        "apparent_temp": apparent_temps,
        "precipitation": precipitation,
        "time_sin": time_sin,
        "time_cos": time_cos,
        "future_temp_change": future_temp_change,
        "target_temp": target_temp,
        "run_time": run_time,  # Run time for heating (minutes)
        "boiler_temp": boiler_temp  # Temperature setpoint for boiler
    })
    
    return df

# Generate data
df = generate_dummy_data(1000)

# Convert to tensor
X = torch.tensor(df.drop(columns=["run_time", "boiler_temp"]).values, dtype=torch.float32)
y = torch.tensor(df[["run_time", "boiler_temp"]].values, dtype=torch.float32)  # Targets are run time and boiler temp

# Define Neural Network
class HeatingPredictor(nn.Module):
    def __init__(self):
        super(HeatingPredictor, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)  # Output: [Run Time (mins), Boiler Temperature]
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.relu(self.fc3(x))  # Ensure non-negative outputs


# Initialize model
model = HeatingPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train Model
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save Model
torch.save(model.state_dict(), "heating_model.pth")

print(model.state_dict())