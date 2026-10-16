"""Train a small PyTorch controller without biases, then verify the loop.

The example controller of nnstab (nnstab.example_controller()) was trained
in a similar way, with 8 neurons per hidden layer. Needs PyTorch
(pip install torch).
"""

import torch
from torch import nn

import nnstab as nst

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(1, 16, bias=False), nn.Tanh(),     # no biases, so pi(0) = 0
                      nn.Linear(16, 16, bias=False), nn.Tanh(),
                      nn.Linear(16, 1, bias=False)).double()

# imitate the controller u = -tanh(2y) - 0.5y on [-3, 3]
y = torch.linspace(-3, 3, 600, dtype=torch.float64).unsqueeze(1)
u = -torch.tanh(2 * y) - 0.5 * y
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(3000):
    opt.zero_grad()
    loss = ((model(y) - u) ** 2).mean()
    loss.backward()
    opt.step()
print(f"training loss: {loss.item():.2e}")
print()

# the plant of the local example, with the new controller
A, B, C, _ = nst.example_system("local")
res = nst.verify(A, B, C, model)                 # the model is read with nnsector
print(res)
