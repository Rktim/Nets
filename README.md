# Nets
 A Lightweight Neural Network Framework (NumPy-Based)

Nets is a from-scratch deep learning framework built using NumPy, designed to be educational, modular, and extensible, while also supporting real-world experimentation workflows such as training, evaluation, and visualization.


## 🔥 Highlights
	•	🧠 Custom autograd engine (Tensor-based)
  
	•	⚙️ Modular neural network API (like PyTorch)
  
	•	📦 Layers: Linear, RNN, (Conv2D in progress)
  
	•	⚡ Optimizers: SGD, Adam
  
	•	📉 Losses: MSE, CrossEntropy
  
	•	📊 Advanced Dashboard (W&B-like)
  
	•	🧪 Experiment tracking system
  
	•	📁 Save / Load run logs
  
	•	🎯 Supports MLP, RNN, and basic CNN workflows


## 🧠 Core Concepts

### Tensor
	•	Wraps NumPy arrays
  
	•	Supports automatic differentiation
  
	•	Builds computational graph dynamically

from nets.tensor.tensor import Tensor
x = Tensor([1,2,3], requires_grad=True)


### Model Definition

from nets.nn import Sequential, Linear, ReLU

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

### Training

from nets.optim.adam import Adam
from nets.losses.cross_entropy import cross_entropy

optimizer = Adam(model.parameters(), lr=0.001)

logits = model(x)
loss = cross_entropy(logits, y)

loss.backward()
optimizer.step()
optimizer.zero_grad()

### 📊 Dashboard (W&B-like)

NETS includes a real-time experiment dashboard powered by Dash + Plotly.

Features:
	•	📈 Live Loss & Accuracy plots
	•	📊 Precision / Recall / F1 (auto-detected)
	•	🔥 Confusion Matrix
	•	🔁 Run comparison
	•	🧾 Experiment metadata
	•	🗑 Run deletion
	•	🎯 Task-aware visualization

### Run Dashboard

from nets.visualization.dashboard import Dashboard
Dashboard(logger).run()



### 📊 Metrics Supported

Classification
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 Score
	•	Confusion Matrix

Regression (planned)
	•	MAE
	•	RMSE
	•	R²

⸻

### 🧩 Features (Detailed)

Autograd Engine
	•	Reverse-mode differentiation
	•	Dynamic graph construction
	•	Supports broadcasting

Layers
	•	Linear
	•	ReLU / Sigmoid / Tanh
	•	RNN
	•	Conv2D (in progress)

Optimizers
	•	SGD (with momentum)
	•	Adam

Data
	•	Dataset abstraction
	•	DataLoader (mini-batch support)

Visualization
	•	Real-time metrics
	•	Interactive plots
	•	Task-aware rendering

⸻

### ⚠️ Limitations
	•	No GPU acceleration (NumPy backend only)
	•	Conv2D still being optimized
	•	No Transformer yet
	•	Limited NLP support (no embeddings)

⸻

### 🚀 Future Work
	•	🔥 Transformer (Encoder + Decoder)
	•	⚡ GPU support (CuPy backend)
	•	📦 Model saving / loading
	•	🧠 Automatic Trainer (Lightning-like)
	•	🌐 Deployment utilities

⸻

### 🤝 Contribution

Feel free to fork and improve:
	•	Add layers (Conv, Attention)
	•	Improve performance
	•	Enhance dashboard UI

⸻

### 📸 Dashboard Preview

Below are real snapshots of the NETS Dashboard during training:

🔹 RNN Training View
<img width="1878" height="666" alt="image" src="https://github.com/user-attachments/assets/3080b8b8-05a8-4920-8a32-aa33fc9bf9d7" />

🔹 MLP MNIST Training View
<img width="1879" height="814" alt="image" src="https://github.com/user-attachments/assets/8f231778-f0a8-41a6-ba1b-2e99b50c4fa2" />

These demonstrate:
	•	Real-time metric tracking
	•	Adaptive visualization (task-aware)
	•	Clean UI with multiple metric panels

⸻

### 🏁 Conclusion

NETS demonstrates that:

You can build a functional deep learning framework from scratch,
understand every component deeply,
and still run meaningful experiments.

⸻

⭐ If you like this project

Give it a star ⭐ and share it!
