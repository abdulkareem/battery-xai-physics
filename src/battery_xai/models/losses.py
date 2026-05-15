"""Physics-aware losses and constraints for temporal degradation models."""
from __future__ import annotations


def monotonic_degradation_penalty(predictions):
    """Penalize SOH increases over cycle order for tensor predictions."""

    import torch

    if predictions.ndim == 1:
        diffs = predictions[1:] - predictions[:-1]
    else:
        diffs = predictions[:, 1:] - predictions[:, :-1]
    return torch.relu(diffs).mean()


def resistance_growth_penalty(predicted_resistance):
    """Penalize predicted internal-resistance decreases."""

    import torch

    diffs = predicted_resistance[1:] - predicted_resistance[:-1]
    return torch.relu(-diffs).mean()


class PhysicsAwareLoss:
    """MSE plus monotonic temporal consistency regularization."""

    def __init__(self, physics_weight: float = 0.1, temporal_weight: float = 0.05) -> None:
        import torch

        self.mse = torch.nn.MSELoss()
        self.physics_weight = physics_weight
        self.temporal_weight = temporal_weight

    def __call__(self, prediction, target):
        import torch

        base = self.mse(prediction, target)
        mono = monotonic_degradation_penalty(prediction.flatten()) if prediction.numel() > 1 else torch.tensor(0.0, device=prediction.device)
        smooth = torch.mean(torch.diff(prediction.flatten(), n=2) ** 2) if prediction.numel() > 2 else torch.tensor(0.0, device=prediction.device)
        return base + self.physics_weight * mono + self.temporal_weight * smooth
