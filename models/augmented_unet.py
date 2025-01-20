import torch
from torch import nn
from torch.nn import functional as F
from segmentation_models_pytorch.base import SegmentationHead, SegmentationModel, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from typing import List, Optional, Union, Any

class AugmentedUnet(SegmentationModel):
    def __init__(
            self,
            encoder_depth: int = 5,
            encoder_name: str = "resnet34",
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: any=None,
            **kwargs: dict[str, Any]
        ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        decoder_in_channels = self.encoder.out_channels[:-1] + (self.encoder.out_channels[-1] + 4, )

        self.decoder = UnetDecoder(
            encoder_channels=decoder_in_channels, # Additional featres from DAC and RMP Block
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.dac = DACBlock(channel=self.encoder.out_channels[-1])
        self.rmp = RMPBlock(channel=self.encoder.out_channels[-1])

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        # Sanity check input shape
        self.check_input_shape(x)

        # Pass the input through the encoder
        features = self.encoder(x)

        # Pass the encoder output through the bottleneck layers
        hires_features = features[:-1] + [self.rmp(self.dac(features[-1]))]
        decoder_output = self.decoder(*hires_features)

        masks = self.segmentation_head(decoder_output)

        return masks

class DACBlock(nn.Module):
    def __init__(self, channel: int):
        super(DACBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        
        self.apply(self.initalize_weights)

    def initalize_weights(self, module):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        dilated1_out = F.relu(self.dilate1(x))
        dilated2_out = F.relu(self.conv1x1(self.dilate2(x)))
        dilated3_out = F.relu(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilated4_out = F.relu(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilated1_out + dilated2_out + dilated3_out + dilated4_out
        return out

class RMPBlock(nn.Module):
    def __init__(self,  channel: int, kernel_sizes: List[int]=[2, 3, 5, 6]):
        super(RMPBlock, self).__init__()
        self.pooling_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=k) for k in kernel_sizes])
        self.conv = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        layers = [
            F.upsample(self.conv(l(x)), size=(h, w), mode='bilinear',align_corners=True) for l in self.pooling_layers
        ]

        layers.append(x)
        out = torch.cat(layers, 1)

        return out