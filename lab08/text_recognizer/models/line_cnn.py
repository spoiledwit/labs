"""Basic building blocks for convolutional models over lines of text."""
import argparse
import math
from typing import Any, Dict, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


# Common type hints
Param2D = Union[int, Tuple[int, int]]

CONV_DIM = 32
FC_DIM = 512
FC_DROPOUT = 0.2
WINDOW_WIDTH = 16
WINDOW_STRIDE = 8


class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Param2D = 3,
        stride: Param2D = 1,
        padding: Param2D = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the ConvBlock to x.

        Parameters
        ----------
        x
            (B, C, H, W) tensor

        Returns
        -------
        torch.Tensor
            (B, C, H, W) tensor
        """
        c = self.conv(x)
        r = self.relu(c)
        return r


class _ResBlock(nn.Module):
    """
    ResNet-style block with skip connection.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with projection if needed
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.skip(identity)
        out += identity
        out = F.relu(out)
        
        return out


class LineCNN(nn.Module):
    """
    Model that uses a simple CNN to process an image of a line of characters with a window, outputs a sequence of logits
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}
        self.num_classes = len(data_config["mapping"])
        self.output_length = data_config["output_dims"][0]

        _C, H, _W = data_config["input_dims"]
        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        fc_dropout = self.args.get("fc_dropout", FC_DROPOUT)
        self.WW = self.args.get("window_width", WINDOW_WIDTH)
        self.WS = self.args.get("window_stride", WINDOW_STRIDE)
        self.limit_output_length = self.args.get("limit_output_length", False)

        # Input is (1, H, W)
        self.convs = self._build_advanced_cnn(conv_dim, fc_dim, H)
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(fc_dim, self.num_classes)

        self._init_weights()

    def _build_advanced_cnn(self, conv_dim, fc_dim, H):
        # ResNet-style blocks with batch normalization
        return nn.Sequential(
            # Initial layer
            nn.Conv2d(1, conv_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            
            # ResNet-style blocks with shortcut connections
            _ResBlock(conv_dim, conv_dim),
            _ResBlock(conv_dim, conv_dim),
            nn.MaxPool2d(2, 2),  # Stride 2 pooling
            
            _ResBlock(conv_dim, conv_dim*2),
            nn.MaxPool2d(2, 2),
            
            _ResBlock(conv_dim*2, conv_dim*4),
            nn.MaxPool2d(2, 2),
            
            _ResBlock(conv_dim*4, conv_dim*8),
            nn.MaxPool2d(2, 2),
            
            # Adaptive feature dimensionality
            nn.Conv2d(conv_dim*8, fc_dim, kernel_size=(H//16, 3), stride=1, padding=(0,1)),
            nn.BatchNorm2d(fc_dim),
            nn.ReLU(),
        )

    def _init_weights(self):
        """
        Initialize weights in a better way than default.
        See https://github.com/pytorch/pytorch/issues/18182
        """
        for m in self.modules():
            if type(m) in {
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            }:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the LineCNN to a black-and-white input image.

        Parameters
        ----------
        x
            (B, 1, H, W) input image

        Returns
        -------
        torch.Tensor
            (B, C, S) logits, where S is the length of the sequence and C is the number of classes
            S can be computed from W and self.window_width
            C is self.num_classes
        """
        _B, _C, _H, _W = x.shape
        x = self.convs(x)  # (B, FC_DIM, 1, Sx)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, S, FC_DIM)
        x = F.relu(self.fc1(x))  # -> (B, S, FC_DIM)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, S, C)
        x = x.permute(0, 2, 1)  # -> (B, C, S)
        if self.limit_output_length:
            x = x[:, :, : self.output_length]
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--fc_dropout", type=float, default=FC_DROPOUT)
        parser.add_argument(
            "--window_width",
            type=int,
            default=WINDOW_WIDTH,
            help="Width of the window that will slide over the input image.",
        )
        parser.add_argument(
            "--window_stride",
            type=int,
            default=WINDOW_STRIDE,
            help="Stride of the window that will slide over the input image.",
        )
        parser.add_argument("--limit_output_length", action="store_true", default=False)
        return parser
