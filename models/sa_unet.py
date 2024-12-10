from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from typing import Optional, Any, Union, List
import torch.nn as nn
import torch

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

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        if not torch.jit.is_tracing():
            self.check_input_shape(x)

        features = self.spatial_attention(self.encoder(x))
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

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