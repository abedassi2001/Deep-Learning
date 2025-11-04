 📊 DATA PREPARATION
python
import torch
import torch.nn as nn
import torch.optim as optim

# Sample 1: Simple Tensor Data (y = 2x + 1)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)

# Sample 2: Random Data (more realistic)
torch.manual_seed(42)
X = torch.randn(100, 5)  # 100 samples, 5 features
y = torch.randn(100, 1)  # 100 targets, 1 output

# Sample 3: Classification Data
X_class = torch.randn(100, 3)  # 100 samples, 3 features
y_class = torch.randint(0, 2, (100,))  # Binary classification (0 or 1)

print(f"Input shape: {X.shape}, Output shape: {y.shape}")
2. 🧩 MODEL DEFINITION
Option A: Simple Linear Model
python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)  # 1 input, 1 output
    
    def forward(self, x):
        return self.layer1(x)

model = SimpleModel()
Option B: Multi-Layer Network
python
class DeepModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input -> Hidden
            nn.ReLU(),                           # Activation
            nn.Linear(hidden_size, output_size)  # Hidden -> Output
        )
    
    def forward(self, x):
        return self.network(x)

# Usage
model = DeepModel(input_size=5, hidden_size=10, output_size=1)
Option C: Classification Model
python
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

# Usage
model = Classifier(input_size=3, num_classes=2)
3. ⚙️ LOSS & OPTIMIZER
For Regression Problems
python
criterion = nn.MSELoss()              # Mean Squared Error
optimizer = optim.SGD(model.parameters(), lr=0.01)
For Classification Problems
python
criterion = nn.CrossEntropyLoss()     # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
Advanced Optimizer Setup
python
criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001  # Regularization
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
4. 🔄 TRAINING LOOP
Basic Training Loop
python
def train_model(model, X, y, epochs=100):
    model.train()  # Set model to training mode
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)
        
        # Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Run training
train_model(model, X, y, epochs=100)
Advanced Training with Validation
python
def train_with_validation(model, X_train, y_train, X_val, y_val, epochs=100):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_pred = model(X_train)
        train_loss = criterion(train_pred, y_train)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
        
        # Track losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses
5. 📈 EVALUATION
Basic Evaluation
python
def evaluate_model(model, X_test, y_test):
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():  # Disable gradient calculation
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        
        # For regression: Calculate additional metrics
        mae = torch.abs(predictions - y_test).mean()
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
    return predictions

# Usage
test_predictions = evaluate_model(model, X_test, y_test)
Classification Evaluation
python
def evaluate_classifier(model, X_test, y_test):
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)  # Get class with highest probability
        accuracy = (predicted == y_test).float().mean()
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Predictions: {predicted}")
        print(f"Actual: {y_test}")
6. 🎯 PREDICTION
Making Predictions on New Data
python
def make_predictions(model, new_data):
    model.eval()
    
    with torch.no_grad():
        # Ensure input is properly formatted
        if isinstance(new_data, list):
            new_data = torch.tensor(new_data, dtype=torch.float32)
        
        predictions = model(new_data)
        return predictions

# Example usage
new_samples = torch.tensor([[5.0], [6.0], [7.0]], dtype=torch.float32)
predictions = make_predictions(model, new_samples)
print(f"Predictions for new data: {predictions}")
🏗️ COMPLETE WORKING EXAMPLE
python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. DATA
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

# 2. MODEL
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

model = SimpleNN()

# 3. LOSS & OPTIMIZER
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. TRAINING
print("Training started...")
for epoch in range(100):
    # Forward
    pred = model(X)
    loss = criterion(pred, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 5. EVALUATION
model.eval()
with torch.no_grad():
    test_pred = model(X)
    test_loss = criterion(test_pred, y)
    print(f"\nFinal Loss: {test_loss:.4f}")

# 6. PREDICTION
new_x = torch.tensor([[5.0]], dtype=torch.float32)
prediction = model(new_x)
print(f"Prediction for input 5.0: {prediction.item():.2f}")
🎯 QUICK REFERENCE TEMPLATE
python
# TEMPLATE: Copy-paste and fill in your specifics

# 1. DATA
# X = your_input_data
# y = your_target_data

# 2. MODEL
# class YourModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Define your layers here
#     def forward(self, x):
#         # Define data flow here
#         return x

# 3. LOSS & OPTIMIZER  
# criterion = nn.YourLossFunction()
# optimizer = optim.YourOptimizer(model.parameters(), lr=0.001)

# 4. TRAINING
# for epoch in range(num_epochs):
#     # Forward, backward, update

# 5. EVALUATION
# model.eval()
# with torch.no_grad():
#     # Calculate performance

# 6. PREDICTION
# new_predictions = model(new_data)