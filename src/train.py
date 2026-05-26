import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Import your architecture
from model import DeepONet

# Data Generation
def generate_mixed_beam_data(num_samples, num_sensors):
    sensors = np.linspace(0, 1, num_sensors)
    u_data, y_data, w_data = [], [], []
    
    for _ in range(num_samples):
        # 1. Randomly pick which family of load to generate
        load_type = np.random.choice(['sine', 'udl', 'triangular'])
        
        # 2. Randomize the amplitude/magnitude
        A = np.random.uniform(-15.0, 15.0)
        
        # 3. Pick a random continuous coordinate for the Trunk
        y = np.random.uniform(0, 1)
        
        # 4. Generate the Sensor Data (Branch) and True Deflection (Trunk)
        if load_type == 'sine':
            q_sensors = A * np.sin(np.pi * sensors)
            w_true = (A / (np.pi**4)) * np.sin(np.pi * y)
            
        elif load_type == 'udl':
            q_sensors = np.full_like(sensors, A)
            w_true = (A * y / 24.0) * (1.0 - 2.0 * y**2 + y**3)
            
        elif load_type == 'triangular':
            q_sensors = A * sensors  # Load increases linearly from 0 to A
            w_true = (A * y / 360.0) * (7.0 - 10.0 * y**2 + 3.0 * y**4)
            
        u_data.append(q_sensors)
        y_data.append([y])
        w_data.append([w_true])
        
    return (torch.tensor(np.array(u_data), dtype=torch.float32), 
            torch.tensor(np.array(y_data), dtype=torch.float32), 
            torch.tensor(np.array(w_data), dtype=torch.float32))

# Setup and Training
num_sensors = 10       
p_features = 100        
num_epochs = 3000
learning_rate = 1e-3

model = DeepONet(num_sensors, p_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Generating 5000 training samples...")
u_train, y_train, w_train = generate_mixed_beam_data(5000, num_sensors)

print("Starting training...")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(u_train, y_train)
    loss = criterion(predictions, w_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 300 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6e}")

# SAVING THE MODEL
os.makedirs('../saved_models', exist_ok=True)
model_filename = "../saved_models/deeponet_beam_model.pth"
torch.save(model.state_dict(), model_filename)
print(f"\nModel weights saved to '{model_filename}'")

# Inference & Error Metrics
print("\nRunning Inference on new continuous domain...")

A_test = 5.0
test_sensors = np.linspace(0, 1, num_sensors) 
test_q = A_test * np.sin(np.pi * test_sensors)

# Shape: [1, 10]
u_test = torch.tensor(test_q, dtype=torch.float32).unsqueeze(0)

# Shape: [100, 1]
y_plot = np.linspace(0, 1, 100)
y_test = torch.tensor(y_plot, dtype=torch.float32).unsqueeze(1)

model.eval()
with torch.no_grad():
    pred_w = model(u_test, y_test).numpy().flatten()

true_w = (A_test / (np.pi**4)) * np.sin(np.pi * y_plot)

# CALCULATING ERROR METRICS 
mse = np.mean((pred_w - true_w)**2)
mae = np.mean(np.abs(pred_w - true_w))
# Relative L2 Error = ||Pred - True||_2 / ||True||_2
l2_rel_error = np.linalg.norm(pred_w - true_w) / np.linalg.norm(true_w)

print("\n--- Error Metrics ---")
print(f"Mean Squared Error (MSE):  {mse:.6e}")
print(f"Mean Absolute Error (MAE): {mae:.6e}")
print(f"Relative L2 Error:         {l2_rel_error * 100:.4f} %")

# Plotting and Saving the Figure
plt.figure(figsize=(10, 6))

plt.plot(y_plot, true_w, 'k-', linewidth=2.5, label='True Deflection (Analytical)')
plt.plot(y_plot, pred_w, 'r--', linewidth=2.5, label='DeepONet Prediction')

plt.scatter(test_sensors, np.zeros_like(test_sensors), color='blue', marker='v', 
            s=100, label='Fixed Sensor Locations (Branch Input)', zorder=5)

plt.title(f"Simply Supported Beam Deflection", fontsize=14)
plt.xlabel("Normalized Coordinate ($x/L$)", fontsize=12)
plt.ylabel("Deflection $w(x)$", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.gca().invert_yaxis()
plt.tight_layout()

# SAVING THE PLOT
os.makedirs('../output_graphs', exist_ok=True)
plot_filename = "../output_graphs/deeponet_deflection_plot.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"\nPlot saved successfully to '{plot_filename}'")

plt.show()
