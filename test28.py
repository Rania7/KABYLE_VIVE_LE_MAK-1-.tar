import torch
x = torch.tensor([1])
x.resize_(50, 1)
print(x)