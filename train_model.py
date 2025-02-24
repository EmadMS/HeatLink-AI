import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the AI model
class BoilerControlNN(nn.Module):
    def __init__(self):
        super(BoilerControlNN, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)  # Output: [Run Time (mins), Boiler Temperature]
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.relu(self.fc3(x))  # Ensure non-negative outputs

# Initialize the model
model = BoilerControlNN()

# Loss function & optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop (with dummy data)
num_epochs = 1000
for epoch in range(num_epochs):
    # Simulated input: [Room Temp, Target Temp, Outside Temp, Apparent Temp, Future Temp Change, Precipitation, Time Sin, Time Cos]
    X_train = torch.tensor([[19.0, 21.5, 5.0, 3.0, -2.0, 0.1, 0.5, -0.5]], dtype=torch.float32)
    Y_train = torch.tensor([[30, 20.0]], dtype=torch.float32)  # Run 30 mins, Set to 20°C

    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Save model
torch.save(model.state_dict(), "instance/boiler_model.pth")
print("Model trained and saved.")
