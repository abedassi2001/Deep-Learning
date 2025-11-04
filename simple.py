import torch
import torch.nn as nn
import torch.optim as optim

# 1️⃣ Create simple data: y = 2x + 3
x = torch.linspace(0, 10, 50).unsqueeze(1)  # shape (50, 1)
y = 2 * x + 3 + torch.randn(x.size()) * 0.5  # add some noise

# 2️⃣ Define a simple NN model (one linear layer)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 3️⃣ Define loss function (MSE) and optimizer (SGD)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4️⃣ Train the model
epochs = 200
for epoch in range(epochs):
    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5️⃣ Show final parameters
[w, b] = model.parameters()
print(f"\nLearned weight: {w.item():.2f}, bias: {b.item():.2f}")
