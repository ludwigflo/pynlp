import torch.nn as nn
import torch


class LinearClassifier(nn.Module):

    def __init__(self, num_features: int) -> None:
        """
        Classifies input data points as linear combination of the data point features, followed by a Sigmoid function.

        Parameters
        ----------
        num_features: Dimensionality of the data points.
        """
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the linear classifier.

        Parameters
        ----------
        x: Input data point (or input data batch).

        Returns
        -------
        predictions: Predictions of the linear classifier for an input data point.
        """

        predictions = self.classifier(x)

        return predictions


class SimpleClassifier(nn.Module):

    def __init__(self, num_features: int, num_hidden_features: int=300) -> None:
        """
        Parameters
        ----------
        num_features: Dimensionality of the data points.
        num_hidden_features: Dimensionality of the hidden Layer.
        """

        super(SimpleClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_hidden_features),
            nn.ReLU(),
            nn.Linear(num_hidden_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the linear classifier.

        Parameters
        ----------
        x: Input data point (or input data batch).

        Returns
        -------
        predictions: Predictions of the linear classifier for an input data point.
        """

        predictions = self.classifier(x)

        return predictions