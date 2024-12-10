import torch
from torch import nn
from segmentation_models_pytorch import Unet
from typing import List, Optional, Union, Any


class StackedUnet(nn.Module):
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
            num_refine_layers: int=2,
            **kwargs: dict[str, Any]
        ):
        super().__init__()
        self.main_unet = Unet(
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

        self.refine_unet = nn.ModuleList([
            Unet(
                encoder_depth=encoder_depth - (i + 2),
                encoder_name=encoder_name,
                encoder_weights=None,
                decoder_use_batchnorm=decoder_use_batchnorm,
                decoder_channels=decoder_channels[(i + 2):],
                decoder_attention_type=decoder_attention_type,
                in_channels=classes,
                classes=classes,
                activation=activation,
                aux_params=aux_params
            ) for i in range(num_refine_layers)
        ])

        self.high_resolution_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=classes,
                out_channels=classes,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=classes,
                out_channels=classes,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=classes,
                out_channels=classes,
                kernel_size= 3, 
                padding=1
            )
        )

    def forward(self, x):
        coarse_mask = self.main_unet(x)
        refine_mask = coarse_mask
        for l in self.refine_unet:
            refine_mask = l(refine_mask) + refine_mask
        
        high_resolution_mask = self.high_resolution_decoder(refine_mask)
        
        return high_resolution_mask