import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Step 1: Generating Deep Telemetry Matrix ---")
np.random.seed(101)
simulated_flights = 800  
altitude_ft = np.random.uniform(1000, 25000, simulated_flights)
airspeed_knots = np.random.uniform(120, 350, simulated_flights)
air_density_ratio = np.exp(-altitude_ft / 23000)
compressor_inlet_press = 14.696 * air_density_ratio
propeller_rpm = np.random.uniform(1700, 2200, simulated_flights)
turbine_inlet_temp_k = np.random.uniform(950, 1350, simulated_flights)
optimal_sweep_deg = np.clip((airspeed_knots * 0.12) + (altitude_ft * 0.001), 0, 45)

X = np.column_stack((altitude_ft, airspeed_knots, compressor_inlet_press, turbine_inlet_temp_k, propeller_rpm))
y = optimal_sweep_deg.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

class HighPrecisionFlightBrain(nn.Module):
    def __init__(self, input_dim):
        super(HighPrecisionFlightBrain, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)         
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 8)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(8, 1)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.output_layer(x)

model = HighPrecisionFlightBrain(input_dim=5)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) 

print("--- Step 2: Training High-Precision AI on Cloud GPU ---")
for epoch in range(400):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

print("--- Step 3: Generating Visual Optimization Chart ---")
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    predicted_angles = test_predictions.numpy().flatten()
    actual_angles = y_test_tensor.numpy().flatten()

sorted_indices = np.argsort(X_test[:, 1])
sorted_airspeed = X_test[sorted_indices, 1]

plt.figure(figsize=(10, 5))
plt.plot(sorted_airspeed, actual_angles[sorted_indices], label='Target Physics Sweep (Ideal)', color='blue', linewidth=2)
plt.scatter(sorted_airspeed, predicted_angles[sorted_indices], label='AI Precision Command (Predicted)', color='red', s=15, alpha=0.8)
plt.title('Aslam Industries: Upgraded Neuro-Adaptive Flight Matrix')
plt.xlabel('Airspeed (Knots)')
plt.ylabel('Wing Sweep Angle (Degrees)')
plt.legend()
plt.grid(True)
plt.show()
