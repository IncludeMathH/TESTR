import torch

a = torch.zeros(1, 6, 3)      # B, N, C
mask = torch.tensor([[1],
                     [0],
                     [1],
                     [1],
                     [0],
                     [0]])
b = a+mask
print(b)
