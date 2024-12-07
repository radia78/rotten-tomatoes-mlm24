from segmentation_models_pytorch import Unet
import torch.nn as nn
import torch

class StackedUnet(nn.Module):
    def __init__(
        self, 
        encoder_name='efficientnet-b3', 
        encoder_depth=5, 
        encoder_weights='imagenet', 
        decoder_use_batchnorm=True, 
        decoder_channels=(256, 128, 64, 32, 16), 
        decoder_attention_type=None, 
        in_channels=3, 
        classes=1,
        activation=None, 
        aux_params=None,
        num_layers=2,
        **kwargs
    ):
        super().__init__()
        self.main_unet = Unet(
            encoder_name=encoder_name, 
            encoder_depth=encoder_depth, 
            encoder_weights=encoder_weights, 
            decoder_use_batchnorm=decoder_use_batchnorm, 
            decoder_channels=decoder_channels, 
            decoder_attention_type=decoder_attention_type, 
            in_channels=in_channels, 
            classes=classes, 
            activation=activation, 
            aux_params=aux_params
        )

        self.refine_layers = nn.ModuleList([
            Unet(
                encoder_name='efficientnet-b0', 
                encoder_depth=encoder_depth - 3, 
                encoder_weights=None, 
                decoder_use_batchnorm=True, 
                decoder_channels=decoder_channels[:-3], 
                decoder_attention_type=decoder_attention_type, 
                in_channels=in_channels + 1, 
                classes=1, 
                activation=None, 
                aux_params=None
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        coarse_mask = self.main_unet(x)
        refine_mask = coarse_mask
        for l in self.refine_layers: 
            refine_mask = torch.cat([x, refine_mask], dim=1)
            refine_mask = l(refine_mask)

        return refine_mask