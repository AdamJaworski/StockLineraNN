import torch
import torch_directml

device = torch_directml.device()
model = torch.nn.Linear(5, 5)

model.to(device)
print("end")