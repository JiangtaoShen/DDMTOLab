import torch
import torch.nn as nn


class Sin(nn.Module):
    """Custom Sin activation function"""

    def forward(self, x):
        return torch.sin(x)


class Swish(nn.Module):
    """Custom Swish activation function: x * sigmoid(x)"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, activation='tanh'):
        """
        Physics-Informed Neural Network

        Args:
            in_dim: Input dimension
            hidden_dim: Number of nodes in hidden layers
            out_dim: Output dimension
            num_layer: Total number of layers (including output layer)
                      e.g., num_layer=3 means 2 hidden layers + 1 output layer
            activation: Activation function type
                       Can be string: 'tanh', 'relu', 'sigmoid', 'sin', 'swish'
                       Or float: [0, 1) -> tanh, [1, 2) -> relu, [2, 3) -> sigmoid,
                                [3, 4) -> sin, [4, 5] -> swish
        """
        super(PINNs, self).__init__()

        # Activation function mapping for string input
        activation_map = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'sin': Sin,
            'swish': Swish
        }

        # Activation function mapping for numerical input
        activation_num_map = {
            0: nn.Tanh,  # [0, 1) -> tanh
            1: nn.ReLU,  # [1, 2) -> relu
            2: nn.Sigmoid,  # [2, 3) -> sigmoid
            3: Sin,  # [3, 4) -> sin
            4: Swish  # [4, 5] -> swish
        }

        # Determine activation function class
        if isinstance(activation, (int, float)):
            # Numerical input: validate range first
            if activation < 0 or activation > 5:
                raise ValueError(f"Numerical activation {activation} out of range. "
                                 f"Expected [0, 5], maps to: 0=tanh, 1=relu, 2=sigmoid, 3=sin, 4=swish")

            # Convert to integer index, handle special case for 5
            activation_idx = int(activation) if activation < 5 else 4
            activation_class = activation_num_map[activation_idx]
        elif isinstance(activation, str):
            # String input
            if activation.lower() not in activation_map:
                raise ValueError(f"Activation '{activation}' not supported. "
                                 f"Available: {list(activation_map.keys())}")
            activation_class = activation_map[activation.lower()]
        else:
            raise TypeError(f"Activation must be string or number, got {type(activation)}")

        # Build network layers
        layers = []
        for i in range(num_layer - 1):
            # First layer: in_dim -> hidden_dim; Other layers: hidden_dim -> hidden_dim
            in_features = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation_class())  # Create new activation instance

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)