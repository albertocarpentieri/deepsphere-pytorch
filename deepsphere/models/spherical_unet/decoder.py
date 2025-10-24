"""Decoder for Spherical UNet.
"""
# pylint: disable=W0221

import torch
from torch import nn

from deepsphere.layers.chebyshev import SphericalChebConv
from deepsphere.models.spherical_unet.utils import SphericalChebBN, SphericalChebBNPool


class SphericalChebBNPoolCheb(nn.Module):
    """Building Block calling a SphericalChebBNPool block then a SphericalCheb.
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, middle_channels, lap, pooling, kernel_size)
        self.spherical_cheb = SphericalChebConv(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        x = self.spherical_cheb(x)
        return x


class SphericalChebBNPoolConcat(nn.Module):
    """Building Block calling a SphericalChebBNPool Block
    then concatenating the output with another tensor
    and calling a SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, out_channels, lap, pooling, kernel_size)
        self.spherical_cheb_bn = SphericalChebBN(in_channels + out_channels, out_channels, lap, kernel_size)

    def forward(self, x, concat_data):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]
            concat_data (:obj:`torch.Tensor`): encoder layer output [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        # pylint: disable=E1101
        x = torch.cat((x, concat_data), dim=2)
        # pylint: enable=E1101
        x = self.spherical_cheb_bn(x)
        return x


class SphericalChebBNPoolConcatFixed(nn.Module):
    """Fixed version of SphericalChebBNPoolConcat that properly handles channel mismatches."""

    def __init__(self, in_channels, out_channels, skip_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): input channels from previous decoder layer.
            out_channels (int): target channels after unpooling.
            skip_channels (int): channels from skip connection.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree.
        """
        super().__init__()
        # Step 1: Unpool + conv to target resolution and channels
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, out_channels, lap, pooling, kernel_size)
        # Step 2: Process concatenated features
        concat_channels = out_channels + skip_channels
        self.spherical_cheb_bn = SphericalChebBN(concat_channels, out_channels, lap, kernel_size)

    def forward(self, x, concat_data):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]
            concat_data (:obj:`torch.Tensor`): encoder layer output [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        # Step 1: Unpool and conv to target channels
        x = self.spherical_cheb_bn_pool(x)
        # Step 2: Concatenate with skip connection
        x = torch.cat((x, concat_data), dim=2)
        # Step 3: Project concatenated features to target channels
        x = self.spherical_cheb_bn(x)
        return x


class Decoder(nn.Module):
    """The decoder of the Spherical UNet.
    """

    def __init__(self, unpooling, laps, kernel_size, depth, output_channels=3, embed_size=64):
        """Initialization.

        Args:
            unpooling (:obj:`torch.nn.Module`): The unpooling object.
            laps (list): List of laplacians.
            kernel_size (int): polynomial degree.
            depth (int): Number of decoding levels.
            output_channels (int): Number of output channels.
            embed_size (int): Base embedding dimension, must match encoder's embed_size.
        """
        super().__init__()
        self.unpooling = unpooling
        self.kernel_size = kernel_size
        self.depth = depth
        self.output_channels = output_channels
        self.embed_size = embed_size
        
        # Use configurable embed_size with scaling pattern:
        # Decoder reverses the encoder pattern: embed_size*X -> embed_size*(X/2)
        # For depth=2: encoder outputs [embed_size*2, embed_size]
        # For depth=3: encoder outputs [embed_size*4, embed_size*2, embed_size]  
        # For depth=4: encoder outputs [embed_size*8, embed_size*4, embed_size*2, embed_size]
        
        if depth == 2:
            # Decoder: embed_size*2 -> embed_size (with skip embed_size), then embed_size -> output
            self.decoder_layers = nn.ModuleList([
                SphericalChebBNPoolConcatFixed(embed_size*2, embed_size, embed_size, laps[1], self.unpooling, self.kernel_size),
                SphericalChebConv(embed_size, output_channels, laps[1], self.kernel_size)  # Final conv, no pooling
            ])
        elif depth == 3:
            # Decoder: embed_size*4 -> embed_size*2 (with skip embed_size*2), embed_size*2 -> embed_size (with skip embed_size), then embed_size -> output
            self.decoder_layers = nn.ModuleList([
                SphericalChebBNPoolConcatFixed(embed_size*4, embed_size*2, embed_size*2, laps[1], self.unpooling, self.kernel_size),
                SphericalChebBNPoolConcatFixed(embed_size*2, embed_size, embed_size, laps[2], self.unpooling, self.kernel_size),
                SphericalChebConv(embed_size, output_channels, laps[2], self.kernel_size)  # Final conv, no pooling
            ])
        elif depth == 4:
            # Decoder: embed_size*8 -> embed_size*4 (with skip embed_size*4), embed_size*4 -> embed_size*2 (with skip embed_size*2), embed_size*2 -> embed_size (with skip embed_size), then embed_size -> output
            self.decoder_layers = nn.ModuleList([
                SphericalChebBNPoolConcatFixed(embed_size*8, embed_size*4, embed_size*4, laps[1], self.unpooling, self.kernel_size),
                SphericalChebBNPoolConcatFixed(embed_size*4, embed_size*2, embed_size*2, laps[2], self.unpooling, self.kernel_size),
                SphericalChebBNPoolConcatFixed(embed_size*2, embed_size, embed_size, laps[3], self.unpooling, self.kernel_size),
                SphericalChebConv(embed_size, output_channels, laps[3], self.kernel_size)  # Final conv, no pooling
            ])
        elif depth == 5:
            # Decoder: embed_size*8 -> embed_size*4 (with skip embed_size*4), embed_size*4 -> embed_size*2 (with skip embed_size*2), embed_size*2 -> embed_size (with skip embed_size), then embed_size -> output
            self.decoder_layers = nn.ModuleList([
                SphericalChebBNPoolConcatFixed(embed_size*16, embed_size*8, embed_size*8, laps[1], self.unpooling, self.kernel_size),
                SphericalChebBNPoolConcatFixed(embed_size*8, embed_size*4, embed_size*4, laps[2], self.unpooling, self.kernel_size),
                SphericalChebBNPoolConcatFixed(embed_size*4, embed_size*2, embed_size*2, laps[3], self.unpooling, self.kernel_size),
                SphericalChebBNPoolConcatFixed(embed_size*2, embed_size, embed_size, laps[4], self.unpooling, self.kernel_size),
                SphericalChebConv(embed_size, output_channels, laps[4], self.kernel_size)  # Final conv, no pooling
            ])
        elif depth == 6:
            # Decoder: embed_size*8 -> embed_size*4 (with skip embed_size*4), embed_size*4 -> embed_size*2 (with skip embed_size*2), embed_size*2 -> embed_size (with skip embed_size), then embed_size -> output
            self.decoder_layers = nn.ModuleList([
                SphericalChebBNPoolConcatFixed(embed_size*32, embed_size*16, embed_size*16, laps[1], self.unpooling, self.kernel_size),
                SphericalChebBNPoolConcatFixed(embed_size*16, embed_size*8, embed_size*8, laps[2], self.unpooling, self.kernel_size),
                SphericalChebBNPoolConcatFixed(embed_size*8, embed_size*4, embed_size*4, laps[3], self.unpooling, self.kernel_size),
                SphericalChebBNPoolConcatFixed(embed_size*4, embed_size*2, embed_size*2, laps[4], self.unpooling, self.kernel_size),
                SphericalChebBNPoolConcatFixed(embed_size*2, embed_size, embed_size, laps[5], self.unpooling, self.kernel_size),
                SphericalChebConv(embed_size, output_channels, laps[5], self.kernel_size)  # Final conv, no pooling
            ])
        else:
            raise ValueError(f"Depth {depth} not supported yet. Please use depth 2, 3, or 4.")
        

    def forward(self, encoder_outputs):
        """Forward Pass.

        Args:
            encoder_outputs (list): List of encoder output tensors from coarsest to finest.

        Returns:
            :obj:`torch.Tensor`: output after forward pass.
        """
        # Start with the coarsest encoder output
        x = encoder_outputs[0]
        
        # Process through intermediate decoder layers (with skip connections)
        for i, layer in enumerate(self.decoder_layers[:-1]):
            skip_connection = encoder_outputs[i + 1]
            x = layer(x, skip_connection)
        
        # Final decoder layer (no skip connection)
        x = self.decoder_layers[-1](x)
        
        # if not self.training:
        #     x = self.softmax(x)
        return x
