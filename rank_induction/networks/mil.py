from typing import Tuple, Literal

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Attention layer"""

    def __init__(self, input_dim: int, temperature: float = 1.0) -> None:
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.temperature = temperature
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        alignment = self.linear(x).squeeze(dim=-1)  # (n, 1) -> (n, )
        attention_weight = torch.softmax(alignment / self.temperature, dim=0)  # (n,)
        return attention_weight


class AttentionBasedFeatureMIL(nn.Module):
    """Feature로부터 forward하는 Attention MIL 모델"""

    def __init__(
        self,
        in_features: int,
        adaptor_dim: int = 256,
        num_classes: int = 2,
        temperature: float = 1.0,
        threshold: float = None,
        return_with: Literal[
            "contribution", "attention_weight", "attention_score"
        ] = "attention_score",
        **kwargs
    ):
        super(AttentionBasedFeatureMIL, self).__init__()
        self.in_features = in_features
        self.adaptor_dim = adaptor_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.threshold = threshold
        self.return_with = return_with

        # Adding an adaptor layer
        self.adaptor = nn.Sequential(
            nn.Linear(in_features, adaptor_dim),
            nn.ReLU(),
            nn.Linear(adaptor_dim, in_features),
        )
        self.attention_layer = AttentionLayer(in_features, self.temperature)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): (1, N, D)

        Returns:
            torch.Tensor: _description_
        """
        if x.ndim == 3:
            x = x.squeeze(0)

        instance_features = self.adaptor(x)
        alignment = self.attention_layer.linear(instance_features).squeeze(
            dim=-1
        )  # (N, )

        attention_weights = torch.softmax(alignment / self.temperature, dim=0)  # (n,)

        if self.threshold is not None:
            n_patches = attention_weights.size(0)
            # threshold is not None인 경우 threshold 처리
            thresholded = attention_weights - (self.threshold / n_patches)
            thresholded = torch.clamp(thresholded, min=0.0)
            nom = thresholded.sum() + 1e-8

            attention_weights = thresholded / nom

        weighted_features = torch.einsum(
            "i,ij->ij", attention_weights, instance_features
        )

        context_vector = weighted_features.sum(axis=0)
        logit = self.classifier(context_vector)

        if self.return_with == "contribution":
            instance_contribution = attention_weights * self.classifier(
                instance_features
            ).squeeze(1)
            return logit, instance_contribution

        if self.return_with == "attention_weight":
            return logit, attention_weights

        if self.return_with == "attention_score":
            return logit, alignment

        return logit, attention_weights
