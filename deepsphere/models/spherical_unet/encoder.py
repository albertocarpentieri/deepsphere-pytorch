"""Encoder for Spherical UNet.
"""
# pylint: disable=W0221
from torch import nn

from deepsphere.layers.chebyshev import SphericalChebConv
from deepsphere.models.spherical_unet.utils import SphericalChebBN, SphericalChebBNPool


class SphericalChebBN2(nn.Module):
    """Building Block made of 2 Building Blocks (convolution, batchnorm, activation).
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree.
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spherical_cheb_bn_1 = SphericalChebBN(in_channels, middle_channels, lap, kernel_size)
        self.spherical_cheb_bn_2 = SphericalChebBN(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_1(x)
        x = self.spherical_cheb_bn_2(x)
        return x


class SphericalChebPool(nn.Module):
    """Building Block with a pooling/unpooling and a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.pooling(x)
        x = self.spherical_cheb(x)
        return x


class Encoder(nn.Module):
    """Encoder for the Spherical UNet.
    """

    def __init__(self, pooling, laps, kernel_size, depth, input_channels=16, embed_size=64):
        """Initialization.

        Args:
            pooling (:obj:`torch.nn.Module`): pooling layer.
            laps (list): List of laplacians.
            kernel_size (int): polynomial degree.
            depth (int): Number of encoding levels.
            input_channels (int): Number of input channels.
            embed_size (int): Base embedding dimension, scales up through the network.
        """
        super().__init__()
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.depth = depth
        self.embed_size = embed_size
        
        # Use configurable embed_size with scaling pattern:
        # input_channels → embed_size//2 → embed_size (first layer)
        # embed_size → embed_size*2, embed_size*2 → embed_size*4, etc. (subsequent layers)
        
        if depth == 2:
            self.encoder_layers = nn.ModuleList([
                SphericalChebBN2(input_channels, embed_size//2, embed_size, laps[depth-1], self.kernel_size),
                SphericalChebBNPool(embed_size, embed_size*2, laps[depth-2], self.pooling, self.kernel_size)
            ])
        elif depth == 3:
            self.encoder_layers = nn.ModuleList([
                SphericalChebBN2(input_channels, embed_size//2, embed_size, laps[depth-1], self.kernel_size),
                SphericalChebBNPool(embed_size, embed_size*2, laps[depth-2], self.pooling, self.kernel_size),
                SphericalChebBNPool(embed_size*2, embed_size*4, laps[depth-3], self.pooling, self.kernel_size)
            ])
        elif depth == 4:
            self.encoder_layers = nn.ModuleList([
                SphericalChebBN2(input_channels, embed_size//2, embed_size, laps[depth-1], self.kernel_size),
                SphericalChebBNPool(embed_size, embed_size*2, laps[depth-2], self.pooling, self.kernel_size),
                SphericalChebBNPool(embed_size*2, embed_size*4, laps[depth-3], self.pooling, self.kernel_size),
                SphericalChebBNPool(embed_size*4, embed_size*8, laps[depth-4], self.pooling, self.kernel_size)
            ])
        elif depth == 5:
            self.encoder_layers = nn.ModuleList([
                SphericalChebBN2(input_channels, embed_size//2, embed_size, laps[depth-1], self.kernel_size),
                SphericalChebBNPool(embed_size, embed_size*2, laps[depth-2], self.pooling, self.kernel_size),
                SphericalChebBNPool(embed_size*2, embed_size*4, laps[depth-3], self.pooling, self.kernel_size),
                SphericalChebBNPool(embed_size*4, embed_size*8, laps[depth-4], self.pooling, self.kernel_size),
                SphericalChebBNPool(embed_size*8, embed_size*16, laps[depth-5], self.pooling, self.kernel_size)
            ])
        elif depth == 6:
            self.encoder_layers = nn.ModuleList([
                SphericalChebBN2(input_channels, embed_size//2, embed_size, laps[depth-1], self.kernel_size),
                SphericalChebBNPool(embed_size, embed_size*2, laps[depth-2], self.pooling, self.kernel_size),
                SphericalChebBNPool(embed_size*2, embed_size*4, laps[depth-3], self.pooling, self.kernel_size),
                SphericalChebBNPool(embed_size*4, embed_size*8, laps[depth-4], self.pooling, self.kernel_size),
                SphericalChebBNPool(embed_size*8, embed_size*16, laps[depth-5], self.pooling, self.kernel_size),
                SphericalChebBNPool(embed_size*16, embed_size*32, laps[depth-6], self.pooling, self.kernel_size)
            ])
        else:
            raise ValueError(f"Depth {depth} not supported yet. Please use depth 2, 3, or 4.")

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            list: List of encoder outputs at each level
        """
        encoder_outputs = []
        current_x = x
        
        # Process through all encoder layers
        for i, layer in enumerate(self.encoder_layers):
            current_x = layer(current_x)
            encoder_outputs.append(current_x)
        
        # Return outputs in reverse order (finest to coarsest)
        return encoder_outputs[::-1]


class EncoderTemporalConv(Encoder):
    """Encoder for the Spherical UNet temporality with convolution.
    """

    def __init__(self, pooling, laps, sequence_length, kernel_size):
        """Initialization.

        Args:
            pooling (:obj:`torch.nn.Module`): pooling layer.
            laps (list): List of laplacians.
            sequence_length (int): The number of images used per sample.
            kernel_size (int): Polynomial degree.
        """
        super().__init__(pooling, laps, kernel_size)
        self.sequence_length = sequence_length
        self.enc_l5 = SphericalChebBN2(
            self.enc_l5.in_channels * self.sequence_length,
            self.enc_l5.in_channels * self.sequence_length,
            self.enc_l5.out_channels,
            laps[5],
            self.kernel_size,
        )
