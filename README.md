
# 🌉 DeepONet for Beam Deflection
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

A physics-informed operator learning framework utilizing a **DeepONet** architecture to predict the continuous deflection curve of a simply supported Euler-Bernoulli beam under arbitrary loading conditions.

Instead of learning a single function mapping, this network learns the *operator* mapping the applied load profile $q(x)$ directly to the deflection solution $w(x)$.

---

# ✨ Features
* **Branch-Trunk Architecture:** The Branch net processes discrete sensor readings of the load $q(x)$, while the Trunk net queries continuous spatial coordinates $y$.
* **Physics-Informed Ansatz:** Enforces hard boundary conditions (simply supported at $x=0$ and $x=L$) by wrapping the raw network output in a mathematical Ansatz: `y * (1.0 - y) * raw_output`.
* **Dynamic Data Generation:** The training script automatically generates highly randomized mixed loading profiles (Sine, Uniform Distributed Load, Triangular) on the fly, eliminating the need for large static datasets.
* **Rapid Generalization:** Capable of instant, mesh-free inference on entirely new load magnitudes and spatial coordinates without retraining.

---

## 📂 Project Structure

```text
DeepOnet-Beam-Deflection/
│
├── saved_models/           # (Ignored) Trained .pth weight files
├── output_graphs/          # (Ignored) Visualization outputs
│
├── src/                    
│   ├── model.py            # BranchNet, TrunkNet, and DeepONet classes
│   ├── train.py            # Data generation, training loop, and validation
│   └── test_udl.py         # Inference script for Uniform Distributed Loads
│
├── .gitignore              
├── requirements.txt        
└── README.md
```

# ⚙️ Installation

Clone the repository:
```bash 
git clone https://github.com/YOUR_USERNAME/DeepONet-Beam-Deflection.git
cd DeepONet-Beam-Deflection
```

Create a virtual environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

# 🚀 Usage

1. Training the Operator
Because the data is generated mathematically using numpy, no external dataset download is required. Simply run the training script. It will generate 5,000 randomized beam loading scenarios, train for 3,000 epochs, and save the weights.

```bash
python src/train.py
```

```Outputs: * saved_models/crack_detector_with_validation_cnn.pth

  output_graphs/training_validation_metrics.png
```

2. Running Inference
To test the model's ability to generalize to an unseen constant load (UDL) across a highly refined continuous coordinate space:

```bash
python src/test_udl.py
```
This will compute the Mean Squared Error (MSE), the Relative L2 Error against the exact analytical solution, and pop up a Matplotlib graph visualizing the predicted curve.

# 🛠️ Built With
* PyTorch - The core Deep Learning framework.
* NumPy - Mathematical generation of analytical physics solutions
* Matplotlib - Visualization of continuous domain results
