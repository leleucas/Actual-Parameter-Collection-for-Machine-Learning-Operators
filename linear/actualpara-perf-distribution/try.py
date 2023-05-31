import torch

in_features = torch.tensor([[2,2,2,2],[1,1,1,1]], dtype=torch.float32)

weight_matrix = torch.tensor([
    [5,5,5],
    [3,3,3],
    [4,4,4],
    [2,2,2]
], dtype=torch.float32)
bias_matrix = torch.tensor([1,2,3], dtype=torch.float32)

out_features = torch.nn.functional.linear(in_features, weight_matrix.t(), bias=None)
print(out_features)
out_features = torch.nn.functional.linear(in_features, weight_matrix.t(), bias=bias_matrix)
print(out_features)
