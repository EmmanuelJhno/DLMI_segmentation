from torch import nn
import torch

from models.backbone.mobilenet import MobileNet3d


class Backbone(nn.Module):
    """
    Base class for backbone models. The forward method should output two feature maps:
    a low level one (with higher resolution) and a high level one (with lower resolution).

    Args:
        in_channels (int): number of channels in the input tensors

    Attributes:
        llf_channels (int): number of channels in the low level features
        hlf_channels (int): number of channels in the high level features
    """
    def __init__(self, in_channels, **kwargs):
        self.in_channels = in_channels
        super(Backbone, self).__init__()

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Extracts features from the input image, represented as a tensor.
        Returns extracted low- and high-level features as a tuple of tensors.

        Arguments:
            x (torch.Tensor): the input tensor, must have size (batch_size, in_channels, width[, height[, depth]])

        Returns:
            (torch.Tensor, torch.Tensor): tuple of tensors: (low_level_features, high_level_features).
        """
        raise NotImplementedError
