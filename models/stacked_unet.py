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
            aux_params: any=None
        ):
        super().__init__()
        self.enhancer = Unet(
            encoder_depth=encoder_depth,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=3,
            activation=activation,
            aux_params=aux_params
        )
        
        self.segmenter = Unet(
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

    def forward(self, x):
        enhanced_img = self.enhancer(x)
        mask = self.segmenter(enhanced_img)

        return mask