from typing import Any, Mapping

import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from core.parts import *
from core.modules import ReverseLayerF


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
            resnet = models.resnet50()
        self.resnet = resnet

        self.create_input_block()
        self.create_encoder()
        self.create_siamese_decoder()
        self.create_segment_decoder()
        self.create_domain_classifier()

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
            in_channels=int(self.encoded_channels[0]/4) + 1,
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
        self.domain_classifier.add_module("d_avgpool", nn.AdaptiveAvgPool2d((1, 1)))
        self.domain_classifier.add_module("d_flat", nn.Flatten())
        self.domain_classifier.add_module("d_dropout", nn.Dropout(p=0.4))

        self.domain_classifier.add_module("d_fc1", nn.Linear(2048, 512))
        self.domain_classifier.add_module("d_bn1", nn.BatchNorm1d(512))
        self.domain_classifier.add_module("d_relu1", nn.ReLU(True))

        self.domain_classifier.add_module("d_out", nn.Linear(512, int(domain_n_classes)))
        self.domain_classifier.add_module("d_softmax", nn.LogSoftmax())

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

    def siamese_forward(self, x=None, y=None, encoded_x=None, encoded_y=None, pools_x=None, pools_y=None):
        assert (x is not None) == (encoded_x is None)
        assert (y is not None) == (encoded_y is None)
        assert (encoded_x is None) == (pools_x is None)
        assert (encoded_y is None) == (pools_y is None)

        # Encode
        if encoded_x is None:
            encoded_x, pools_x = self.encode(x)
            encoded_y, pools_y = self.encode(y)

        # Bridge
        a = torch.cat([encoded_x, encoded_y], 1)
        a = self.siamese_bridge(a)

        # Decode
        pools = self.siamese_fuse(pools_x, pools_y)
        a = self.siamese_decode(a, pools)
        a = self.siamese_decoder_out(a)

        return a

    def segment_decode(self, x, encoder_pools):
        for i, block in enumerate(self.segment_decoder, 1):
            key = f"layer_{5 - i}"
            x = block(x, encoder_pools[key])

        return x

    def domain_classify(self, x=None, encoded_feature=None, alpha=0.5):
        assert (x is not None) or (encoded_feature is not None)
        if encoded_feature is None:
            x, pools = self.encode(x)
        else:
            x = encoded_feature
        x = self.segment_bridge(x)

        x = ReverseLayerF.apply(x, alpha)
        return self.domain_classifier(x)

    def segment_forward(self, x, domain_classify=True, alpha=None, pools=None, return_pool=False):
        """
        img_features: [batch_size, channels, width, height]
        cm: [batch_size, 1, width, height]
        """
        if pools is None:
            x, pools = self.encode(x)

        a = self.segment_bridge(x)
        a = self.segment_decode(a, pools)
        a = self.segment_decoder_out(a)

        pools['layer_0'] = a[:, 0:1, ...]

        if domain_classify:
            d = self.domain_classify(encoded_feature=x, alpha=alpha)
            if return_pool:
                return a, d, x, pools
            else:
                return a, d

        if return_pool:
            return a, x, pools
        return a

    def forward(self, x, y):
        x, encoded_x, pools_x = self.segment_forward(x, domain_classify=False, return_pool=True)
        y, encoded_y, pools_y = self.segment_forward(y, domain_classify=False, return_pool=True)

        cm = self.siamese_forward(encoded_x=encoded_x, encoded_y=encoded_y, pools_x=pools_x, pools_y=pools_y)

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

    def change_segmentation_branch_trainable(self, trainable=True):
        self.change_encoder_trainable(trainable)

        for m in self.segment_bridge.parameters():
            m.requires_grad = False

        for i, block in enumerate(self.segment_decoder):
            for m in block.parameters():
                m.requires_grad = trainable

        for param in self.segment_decoder_out.parameters():
            param.requires_grad = trainable

    def load_segmentation_weight(self, state_dicts: Mapping[str, Any]):
        for k in list(state_dicts.keys()):
            if k.startswith("siamese"):
                del state_dicts[k]

        self.load_state_dict(state_dicts, strict=False)

    def get_siamese_parameter(self):
        params = []
        for i, block in enumerate(self.siamese_decoder, 0):
            for n in block.children():
                # print(n)
                params = params + list(n.parameters())

        params = params + list(self.siamese_bridge.parameters())
        params = params + list(self.siamese_decoder_out.parameters())

        return params

    def get_segmentation_parameter(self):
        params = []
        for i, block in enumerate(self.segment_decoder, 0):
            params = params + list(block.parameters())

        params = params + list(self.segment_bridge.parameters())
        params = params + list(self.segment_decoder_out.parameters())

        return params

    def get_encoder_parameter(self):
        params = []
        params = params + list(self.input_block.parameters())
        params = params + list(self.encoder.parameters())

        return params