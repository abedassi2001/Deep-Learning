# 🧠 Neural Network Guide — PyTorch (All-in-One)

This guide explains how to build, train, and evaluate a neural network step-by-step in PyTorch.

---

## 1. Import Libraries
```python
import torch
import torch.nn as nn
import torch.optim as optim
torch: main PyTorch package

torch.nn: contains layers and models

torch.optim: optimization algorithms

2. Prepare Data
python
Copy code
X = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[2.0],[4.0],[6.0],[8.0]])
Neural networks learn patterns between input → output

For real datasets: normalize, preprocess, split into train/test

3. Define the Model
python
Copy code
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)  # input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)

model = SimpleNN()
nn.Linear(in_features, out_features): fully connected layer

forward(): defines how input flows through the network

For deeper networks, you can use:

python
Copy code
self.net = nn.Sequential(
    nn.Linear(10,50),
    nn.ReLU(),
    nn.Linear(50,1)
)
4. Define Loss & Optimizer
python
Copy code
criterion = nn.MSELoss()                 # regression loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # optimizer
Loss function measures how wrong predictions are

Optimizer updates model weights to minimize loss

Other optimizers: Adam, RMSprop, etc.

5. Training Loop
python
Copy code
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    
    # Compute loss
    loss = criterion(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()  # clear old gradients
    loss.backward()        # compute new gradients
    
    # Update weights
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
Steps: forward → compute loss → backward → update → repeat

optimizer.zero_grad() clears previous gradients

loss.backward() computes new gradients

6. Evaluate the Model
python
Copy code
with torch.no_grad():  # disables gradient tracking
    predictions = model(X)
    print(predictions)
Use torch.no_grad() during evaluation for efficiency

Check how well the network learned

7. Activation Functions (Optional)
Non-linearities help model complex patterns

Common activations:

nn.ReLU(): most common, avoids vanishing gradients

nn.Sigmoid(): outputs 0 → 1

nn.Tanh(): outputs -1 → 1

nn.Softmax(dim=1): multi-class classification

Example with Sequential:

python
Copy code
self.net = nn.Sequential(
    nn.Linear(2,16),
    nn.ReLU(),
    nn.Linear(16,1)
)
8. Typical Neural Network Structures
Regression: Input → Linear → ReLU → Linear → Output

Classification: Input → Linear → ReLU → Linear → Softmax

9. Quick Checklist
Import libraries

Prepare data

Define model (layers + forward pass)

Choose loss function

Choose optimizer

Training loop (forward → loss → backward → update)

Evaluate model

(Optional) Add activations or more layers

10. Tips
Normalize your data

Use model.train() / model.eval() when switching between training and testing

Start simple; increase complexity gradually

Watch for overfitting

Experiment with learning rates and optimizers

11. Full Example
python
Copy code
import torch
import torch.nn as nn
import torch.optim as optim

# Data
X = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[2.0],[4.0],[6.0],[8.0]])

# Model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)

model = SimpleNN()

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Evaluation
with torch.no_grad():
    predictions = model(X)
    print("\nPredictions:")
    print(predictions)
pgsql
Copy code

---

This is **fully organized, copy-paste-ready**, and contains **all explanations, code, and tips** in one Markdown file.  

If you want, I can also make a **super compact “cheat sheet version”** that’s literally **<50 lines**