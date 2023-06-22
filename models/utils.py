import torch
import torch.nn as nn
import math


def build_mlp(input_dim, hidden_dim, output_dim, layers):
    """Create a MLP from the configurations"""

    seq = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


class ResidualMLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, num_cond_features=128, num_layers=4):
        super(ResidualMLP, self).__init__()
        self.num_layers = num_layers
        self.init_layer = nn.Sequential(nn.Linear(input_size + num_cond_features, hidden_size), nn.ReLU())
        self.res_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(self.num_layers)])
        self.last_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x, y):
        x_cat = torch.cat([x, y], dim=-1)
        out = self.init_layer(x_cat)
        for i in range(self.num_layers):
            out = self.res_layers[i](out)
        out = self.last_layer(out)
        x += out  # Residual connection
        return x


def vector_to_l_cholesky(z):
    D = (1.0 + math.sqrt(1.0 + 8.0 * z.shape[-1])) / 2.0
    if D % 1 != 0:
        raise ValueError("Correlation matrix transformation requires d choose 2 inputs")
    D = int(D)
    x = torch.zeros(z.shape[:-1] + (D, D), dtype=z.dtype, device=z.device)

    x[..., 0, 0] = 1
    x[..., 1:, 0] = z[..., : (D - 1)]
    i = D - 1
    last_squared_x = torch.zeros(z.shape[:-1] + (D,), dtype=z.dtype, device=z.device)
    for j in range(1, D):
        distance_to_copy = D - 1 - j
        last_squared_x = last_squared_x[..., 1:] + x[..., j:, (j - 1)].clone() ** 2
        x[..., j, j] = (1 - last_squared_x[..., 0]).sqrt()
        x[..., (j + 1) :, j] = z[..., i : (i + distance_to_copy)] * (1 - last_squared_x[..., 1:]).sqrt()
        i += distance_to_copy
    return x
