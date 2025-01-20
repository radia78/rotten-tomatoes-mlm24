import torch
from torch import nn
from torch.nn import functional as F
from segmentation_models_pytorch import Unet
from typing import List, Optional, Union, Any
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.base import SegmentationHead, SegmentationModel, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

'''Augmented UNet'''
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

    def forward(self, x, targets=None):
        # Sanity check input shape
        self.check_input_shape(x)

        # Pass the input through the encoder
        features = self.encoder(x)

        # Pass the encoder output through the bottleneck layers
        hires_features = features[:-1] + [self.rmp(self.dac(features[-1]))]
        decoder_output = self.decoder(*hires_features)

        # Generate the prediction and mask
        masks = self.segmentation_head(decoder_output)
        loss = F.binary_cross_entropy_with_logits(masks, targets) if targets is not None else None

        return masks, loss

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

'''Spatial Attention UNet'''
class SAUnet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(
            encoder_name,
            encoder_depth,
            encoder_weights,
            decoder_use_batchnorm,
            decoder_channels,
            decoder_attention_type,
            in_channels,
            classes,
            activation,
            aux_params,
            **kwargs
        )

        self.spatial_attention = SpatialGate()

    def forward(self, x, targets=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        if not torch.jit.is_tracing():
            self.check_input_shape(x)

        features = self.spatial_attention(self.encoder(x))
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        loss = F.binary_cross_entropy_with_logits(masks, targets) if targets is not None else None
        
        return masks, loss

def channel_pool(x):
    return torch.cat(
        (torch.amax(x, dim=1).unsqueeze(1), torch.mean(x, dim=1).unsqueeze(1)), dim=1
    )
    
class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3

        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv1x1(channel_pool(x))) * x

'''Recurrent Refinement UNet'''
class RRUnet(nn.Module):
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
            k: int = 3
        ):
        super().__init__()

        self.base_network = Unet(
            encoder_depth=encoder_depth,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            aux_params=aux_params
        )
        
        self.refiner = Unet(
            encoder_depth=encoder_depth,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=classes,
            classes=classes,
            activation=activation,
            aux_params=aux_params
        )

        # Set up the number of iterations/refinement
        self.k = k
        self.norm_const = (k * (k + 1)) // 2

    def forward(self, x, targets=None):
        t0 = self.base_network(x)
        t_minus_one = t0
        loss = None if targets is None else F.binary_cross_entropy_with_logits(t0, targets)

        # Gives the final prediction and the total weighted loss over all refinements
        for i in range(self.k):
            t = self.refiner(t_minus_one)
            if targets is not None:
                loss += ((i + 1) / self.norm_const) * F.binary_cross_entropy_with_logits(t, targets)
        
        return t, loss