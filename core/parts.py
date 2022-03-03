import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose", with_down_x=True):
        super().__init__()

        self.with_down_x = with_down_x

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)

        diffY = down_x.size()[-2] - x.size()[-2]
        diffX = down_x.size()[-1] - x.size()[-1]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, down_x], 1)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UnetEncoder(nn.Module):
    def __init__(self, input_channels=3, resnet=None, depth=6, save_output=False):
        super(UnetEncoder, self).__init__()
        # Resnet
        if resnet == None:
            resnet = models.resnet50(pretrained=True)

        self.depth = depth
        self.input_channels = input_channels
        self.save_output = save_output

        # Input block
        input_blocks = list(resnet.children())[:3]

        if input_blocks[0].in_channels != input_channels:
            input_blocks[0] = nn.Conv2d(input_channels, 64, padding=3, kernel_size=7, stride=2)

        self.input_block = nn.Sequential(OrderedDict([
            ('conv1', input_blocks[0]),
            ('bn1', input_blocks[1]),
            ('relu', input_blocks[2])
        ]))

        # Input pool
        self.input_pool = list(resnet.children())[3]

        # Down blocks
        down_blocks = []
        self.encoded_out_channels = []

        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
                out_channel = getattr(bottleneck[-1], 'conv3', bottleneck[-1].conv2)
                self.encoded_out_channels.append(out_channel.out_channels)

        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        """
        Encode an input image using resnet50
        Return:
            x: encoded feature
            pre_pools: saved features during forward
        """
        out = self.input_block(x)

        if self.save_output:
            pre_pools = dict()
            pre_pools[f"layer_0"] = x
            pre_pools[f"layer_1"] = out

        out = self.input_pool(out)

        for i, block in enumerate(self.down_blocks, 2):
            out = block(out)
            if i != (self.depth - 1) and self.save_output:
                pre_pools[f"layer_{i}"] = out

        if self.save_output:
            return out, pre_pools

        return out