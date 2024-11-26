import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.v2 import (
    RandomResizedCrop, 
    RandomRotation, 
    ColorJitter, 
    RandomGrayscale, 
    GaussianBlur, 
    RandomSolarize,
    RandomApply,
    Normalize, 
    Compose, 
    InterpolationMode 
)
from segmentation_models_pytorch.base import SegmentationHead, SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from typing import List, Optional, Union, Any

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class CrossCorrelationLoss(nn.Module):
    def __init__(self, lambd: float=0.2):
        super(CrossCorrelationLoss).__init__()
        self.lambd = lambd

    def forward(self, emb1, emb2):
        c = emb1.T @ emb2
        on_diag = torch.diag(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        return loss

class BarlowTwinsUnet(SegmentationModel):
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
            **kwargs: dict[str, Any]
        ):
        super(BarlowTwinsUnet).__init__()

        # SSL training components
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), # (N, C, H, W) -> (N, C, 1, 1)
            nn.Linear(self.encoder.out_channels + 4, self.encoder.out_channels * 2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(self.encoder.out_channels * 2, affine=False),
            nn.Linear(self.encoder.out_channels * 2, self.encoder.out_channels, bias=None)
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
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

        self.augment_img = Compose([
            RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            RandomApply(
                [RandomRotation((0, 360)), ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            RandomSolarize(p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dac = DACBlock(channel=self.encoder.out_channels)
        self.rmp = RMPBlock(channel=self.encoder.out_channels)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.initialize()

    def generate_embedding(self, x):
        # Sanity check input shape
        self.check_input_shape(x)

        # Pass the input through the encoder
        features = self.encoder(x)

        # Pass the encoder output through the bottleneck layers
        hires_features = self.rmp(self.dac(features))

        # Pass the hi-res features into the projection head
        emb = self.projection_head(hires_features)

        return emb
    
    def __ssl_forward__(self, x):
        x1, x2 = self.augment_img(x), self.augment_img(x)
        z1, z2 = self.generate_embedding(x1), self.generate_embedding(x2)

        return z1.flatten(start_dim=2), z2.flatten(start_dim=2)

    def __unet_forward__(self, x):
        # Sanity check input shape
        self.check_input_shape(x)

        # Pass the input through the encoder
        features = self.encoder(x)

        # Pass the encoder output through the bottleneck layers
        hires_features = self.rmp(self.dac(features))
        decoder_output = self.decoder(*hires_features)

        masks = self.segmentation_head(decoder_output)

        return masks
    
    def forward(self, x, ssl_training=False):
        if ssl_training:
            z1, z2 = self.__ssl_forward__(x)
            return z1, z2
        
        else:
            masks = self.__unet_forward__(x)
            return masks

class DACBlock(nn.Module):
    def __init__(self, channel: int):
        super(DACBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        
        self.apply(self.initalize_weights)

    def initalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilated1_out = nn.ReLU(self.dilate1(x))
        dilated2_out = nn.ReLU(self.conv1x1(self.dilate2(x)))
        dilated3_out = nn.ReLU(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilated4_out = nn.ReLU(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
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
            F.upsample(self.conv(l(x)), size=(h, w), mode='bilinear', align_corners=True) for l in self.pooling_layers
        ]

        layers.append(x)
        out = torch.cat(self.layers, 1)

        return out