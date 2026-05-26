import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import model from the other file
from model import DeepONet


#  Load the Pre-Trained Model
num_sensors = 10
p_features = 100

# Initialize the empty skeleton
model = DeepONet(num_sensors, p_features).to(device)
model_path = "../saved_models/deeponet_beam_model.pth" 

# Load the saved weights into the skeleton
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

# Set the model to evaluation mode (turns off dropout/batchnorm if you had them)
model.eval()

#  Generate a UDL Test Case
# Let's apply a constant uniform load of 10.0 across the whole beam
q_0 = 10.0
print(f"\nGenerating new UDL test case with constant load: {q_0}")

test_sensors = np.linspace(0, 1, num_sensors)
# Create an array where every sensor reads the exact same constant load
test_q = np.full_like(test_sensors, q_0)

# Shape: [1, 10]
u_test = torch.tensor(test_q, dtype=torch.float32).unsqueeze(0)

# Create 100 finely spaced continuous coordinates
y_plot = np.linspace(0, 1, 100)
y_test = torch.tensor(y_plot, dtype=torch.float32).unsqueeze(1)

# ==========================================
# 4. Run Fast Inference
# ==========================================
with torch.no_grad():
    pred_w = model(u_test, y_test).numpy().flatten()

# Calculate the exact analytical solution for a UDL
true_w = (q_0 * y_plot / 24.0) * (1.0 - 2.0 * y_plot**2 + y_plot**3)

# Error Metrics & Plotting
mse = np.mean((pred_w - true_w)**2)
l2_rel_error = np.linalg.norm(pred_w - true_w) / np.linalg.norm(true_w)

print("\n--- Inference Results ---")
print(f"Mean Squared Error: {mse:.6e}")
print(f"Relative L2 Error:  {l2_rel_error * 100:.4f} %")

plt.figure(figsize=(10, 6))
plt.plot(y_plot, true_w, 'k-', linewidth=2.5, label='True Deflection (UDL Analytical)')
plt.plot(y_plot, pred_w, 'r--', linewidth=2.5, label='DeepONet Prediction')

# Visualizing the constant load input
plt.scatter(test_sensors, np.zeros_like(test_sensors), color='blue', marker='v', 
            s=100, label='Sensor Inputs (Constant)', zorder=5)

plt.title(f"DeepONet Generalization Test: UDL $q(x) = {q_0}$", fontsize=14)
plt.xlabel("Normalized Coordinate ($x/L$)", fontsize=12)
plt.ylabel("Deflection $w(x)$", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
