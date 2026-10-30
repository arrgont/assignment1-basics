import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_values = x.max(dim=dim, keepdim=True).values

    shifted = x - max_values

    exp_values = torch.exp(shifted)

    return exp_values / exp_values.sum(
        dim=dim,
        keepdim=True,
    )