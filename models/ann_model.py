"""Small feed-forward ANN for binary student-result classification.

The network returns a raw logit. Pair it with ``torch.nn.BCEWithLogitsLoss``
for training; apply sigmoid outside the model only when a probability is needed.
"""

import torch
from torch import nn


class StudentPerformanceANN(nn.Module):
    """Three-feature binary classifier with one eight-unit hidden layer."""

    def __init__(self) -> None:
        super().__init__()
        # Map the three standardized inputs to eight learned representations.
        self.hidden = nn.Linear(in_features=3, out_features=8)
        # ReLU adds non-linearity while keeping this introductory model simple.
        self.activation = nn.ReLU()
        # Produce one raw score (logit); no sigmoid is applied in the model.
        self.output = nn.Linear(in_features=8, out_features=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one raw logit per input row."""
        hidden_values = self.activation(self.hidden(inputs))
        return self.output(hidden_values)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that receive gradient updates during training."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = StudentPerformanceANN()
    sample_input = torch.tensor([[0.25, -0.50, 1.00]], dtype=torch.float32)
    with torch.no_grad():
        sample_output = model(sample_input)

    print("Model architecture:")
    print(model)
    print(f"Trainable parameters: {count_trainable_parameters(model)}")
    print(f"Sample input shape: {tuple(sample_input.shape)}")
    print(f"Sample output shape: {tuple(sample_output.shape)}")
    print(f"Sample raw logit: {sample_output.item():.6f}")
    if tuple(sample_output.shape) != (1, 1):
        raise RuntimeError(f"Expected output shape (1, 1), got {tuple(sample_output.shape)}")
    print("Model execution: successful; output is a raw logit.")
