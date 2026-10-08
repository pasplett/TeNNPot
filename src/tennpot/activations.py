"""Activation functions supported by TeNNPot model configurations."""

import torch.nn as nn


def get_activation(name):
    """Create an activation from its configuration name."""
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "gelu-speed": GeluSpeed,
    }
    return activations[name.lower()]()


class GeluSpeed(nn.Module):
    """Sigmoid approximation to GELU used by the published model."""

    def forward(self, x):
        """Apply the approximation elementwise."""
        return x * nn.functional.sigmoid(1.702 * x)
