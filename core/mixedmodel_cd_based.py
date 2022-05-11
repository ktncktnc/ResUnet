import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from core.parts import *


class DependentResUnetMultiDecoder(nn.Module):
    def __init__(self, input_channel=3, segment_o_channel=2, cd_o_channel=1, resnet=None):
        super().__init__()
        self.domain_classifier = None
        self.input_pool = None
        self.input_block = None
        self.encoder = None
        self.encoded_channels = None

        self.segment_decoder_out = None
        self.segment_decoder = None
        self.segment_bridge = None

        self.siamese_decoder_out = None
        self.siamese_fusing_blocks = None
        self.siamese_decoder = None
        self.siamese_bridge = None

        self.input_channel = input_channel
        self.segment_o_channel = segment_o_channel
        self.cd_o_channel = cd_o_channel

        if resnet is None:
            resnet = models.resnet34()
        self.resnet = resnet

        self.create_input_block()
        self.create_encoder()
        self.create_siamese_decoder()
        self.create_segment_decoder()

    def create_input_block(self):
        # Input block
        input_blocks = list(self.resnet.children())[:3]
        if input_blocks[0].in_channels != self.input_channel:
            input_blocks[0] = nn.Conv2d(self.input_channel, 64, padding=3, kernel_size=7, stride=2)

        self.input_block = nn.Sequential(OrderedDict([
            ('conv1', input_blocks[0]),
            ('bn1', input_blocks[1]),
            ('relu', input_blocks[2])
        ]))
        self.input_pool = list(self.resnet.children())[3]

    def create_encoder(self):
        encoder = []
        self.encoded_channels = []

        for bottleneck in list(self.resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                encoder.append(bottleneck)

                out_channel = getattr(bottleneck[-1], 'conv3', bottleneck[-1].conv2)
                self.encoded_channels.append(out_channel.out_channels)

        self.encoder = nn.ModuleList(encoder)

    def create_siamese_decoder(self):
        self.siamese_bridge = Bridge(self.encoded_channels[-1] * 2, self.encoded_channels[-1])

        siamese_decoder = []
        siamese_fusing_blocks = []

        for i in range(1, len(self.encoded_channels)):
            siamese_fusing_blocks.append(
                Bridge(self.encoded_channels[-(i + 1)] * 2, self.encoded_channels[-(i + 1)]))

            siamese_decoder.append(
                UpBlockForUNetWithResNet50(self.encoded_channels[-i], self.encoded_channels[-(i + 1)]))

        # Add 2 more blocks

        # Working with input blocks
        siamese_decoder.append(UpBlockForUNetWithResNet50(
            in_channels=int(self.encoded_channels[0]/2) + int(self.encoded_channels[0]/4),
            out_channels=int(self.encoded_channels[0]/2),
            up_conv_in_channels=int(self.encoded_channels[0]),
            up_conv_out_channels=int(self.encoded_channels[0]/2)
        ))

        # Working with input img
        siamese_decoder.append(UpBlockForUNetWithResNet50(
            in_channels=int(self.encoded_channels[0]/4) + self.input_channel,
            out_channels=int(self.encoded_channels[0]/4),
            up_conv_in_channels=int(self.encoded_channels[0]/2),
            up_conv_out_channels=int(self.encoded_channels[0]/4)
        ))

        self.siamese_decoder = nn.ModuleList(siamese_decoder)

        siamese_fusing_blocks.append(Bridge(self.encoded_channels[0] * 2, self.encoded_channels[0]))
        siamese_fusing_blocks.append(Bridge(self.input_channel * 2, self.input_channel))

        self.siamese_fusing_blocks = nn.ModuleList(siamese_fusing_blocks)

        self.siamese_decoder_out = nn.Sequential(
            nn.Conv2d(int(self.encoded_channels[0]/4), self.cd_o_channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def create_segment_decoder(self):
        self.segment_bridge = Bridge(self.encoded_channels[-1], self.encoded_channels[-1])

        segment_decoder = []
        for i in range(1, len(self.encoded_channels)):
            segment_decoder.append(
                UpBlockForUNetWithResNet50(self.encoded_channels[-i], self.encoded_channels[-(i + 1)]))

        segment_decoder.append(UpBlockForUNetWithResNet50(
            in_channels=int(self.encoded_channels[0]/2) + int(self.encoded_channels[0]/4),
            out_channels=int(self.encoded_channels[0]/2),
            up_conv_in_channels=int(self.encoded_channels[0]),
            up_conv_out_channels=int(self.encoded_channels[0]/2)
        ))

        # Working with input img
        segment_decoder.append(UpBlockForUNetWithResNet50(
            in_channels=int(self.encoded_channels[0]/4) + self.input_channel,
            out_channels=int(self.encoded_channels[0]/4),
            up_conv_in_channels=int(self.encoded_channels[0]/2),
            up_conv_out_channels=int(self.encoded_channels[0]/4)
        ))

        self.segment_decoder = nn.ModuleList(segment_decoder)
        self.segment_decoder_out = nn.Sequential(
            nn.Conv2d(int(self.encoded_channels[0]/4), self.segment_o_channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def create_domain_classifier(self, domain_n_classes=2):
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module(nn.Flatten())
        self.domain_classifier.add_module(nn.Linear(int(self.encoded_channels[0]/4), int(self.encoded_channels[0]/8)))
        self.domain_classifier.add_module(nn.Linear(int(self.encoded_channels[0]/8), int(self.encoded_channels[0]/16)))
        self.domain_classifier.add_module(nn.Linear(int(self.encoded_channels[0]/16), int(self.encoded_channels[0]/32)))
        self.domain_classifier.add_module(nn.Linear(int(self.encoded_channels[0]/32), int(domain_n_classes)))

    def input_process(self, x):
        x1 = self.input_block(x)
        x2 = self.input_pool(x1)

        return x2, x1

    def encode(self, x):
        pools = dict()
        pools[f"layer_0"] = x

        x2, x1 = self.input_process(x)
        pools[f"layer_1"] = x1

        for i, block in enumerate(self.encoder, 2):
            x2 = block(x2)
            if i == 5:
                continue
            pools[f"layer_{i}"] = x2

        return x2, pools

    def siamese_fuse(self, pools_x, pools_y):
        pools = dict()
        for i, block in enumerate(self.siamese_fusing_blocks, 1):
            key = f"layer_{5 - i}"
            f = torch.abs(pools_x[key] - pools_y[key])
            pools[key] = f
        return pools

    def siamese_decode(self, fused_x, encoder_pools):
        for i, block in enumerate(self.siamese_decoder, 1):
            key = f"layer_{5 - i}"
            fused_x = block(fused_x, encoder_pools[key])
        return fused_x

    def siamese_forward(self, x, y):
        # Encode
        x, encoder_pools_x = self.encode(x)
        y, encoder_pools_y = self.encode(y)

        # Bridge
        a = torch.cat([x, y], 1)
        a = self.siamese_bridge(a)

        # Decode
        pools = self.siamese_fuse(encoder_pools_x, encoder_pools_y)
        a = self.siamese_decode(a, pools)
        a = self.siamese_decoder_out(a)

        return {
            "x": x,
            "pools_x": encoder_pools_x,
            "y": y,
            "pools_y": encoder_pools_y,
            "cm": a
        }

    def segment_decode(self, x, encoder_pools):
        for i, block in enumerate(self.segment_decoder, 1):
            key = f"layer_{5 - i}"
            x = block(x, encoder_pools[key])

        return x

    def domain_classification(self, x):
        return self.domain_classifier(x)

    def segment_forward(self, x, domain_classify=True, pools=None, cm=None):
        """
        img_features: [batch_size, channels, width, height]
        cm: [batch_size, 1, width, height]
        """
        assert (pools is None) == (cm is None)
        if pools is None:
            x, pools = self.encode(x)
        else:
            pools['layer_0'] = torch.cat([pools['layer_0'], cm], 1)
        x = self.segment_bridge(x)
        a = self.segment_decode(x, pools)
        a = self.segment_decoder_out(a)

        if domain_classify:
            d = self.domain_classification(x)
            return a, d
        return a

    def forward(self, x, y):
        output = self.siamese_forward(x, y)
        cm = output['cm']

        # detached_cm = cm.detach().clone()
        x = self.segment_forward(output['x'], output['pools_x'], cm)
        y = self.segment_forward(output['y'], output['pools_y'], cm)

        return {
            "cm": cm,
            "x": x,
            "y": y
        }

    def change_encoder_trainable(self, trainable=True):
        for param in self.input_block.parameters():
            param.requires_grad = trainable

        for param in self.input_pool.parameters():
            param.requires_grad = trainable

        for block in enumerate(self.encoder):
            for a in block[1]:
                for b in a.children():
                    for param in b.parameters():
                        param.require_grad = trainable
