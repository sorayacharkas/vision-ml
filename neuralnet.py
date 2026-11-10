import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork().to(device)
print(model)      

x = torch.rand(1, 28, 28, device=device)
logits = model(x)
predictions = nn.Softmax(dim=1)(logits)
y_pred = predictions.argmax(1)
print(f"Predicted class: {y_pred}")

input_image = torch.rand(1, 28, 28, device=device)  
print(input_image.size())

#autograd
x = torch.ones(2, 2, requires_grad=True)
y =torch.zeros(2, 2, requires_grad=True)
w = torch.rand(2, 2, requires_grad=True)
b = torch.rand(2, 2, requires_grad=True)
z = torch.matmul(x, w) + b

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f"Loss: {loss}")
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

#computing gradients
loss.backward()
print(f"dL/dw: {w.grad}")
print(f"dL/db: {b.grad}")
