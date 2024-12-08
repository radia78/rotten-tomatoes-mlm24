import torch
from torch import nn
from segmentation_models_pytorch.base import SegmentationHead, SegmentationModel, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from typing import List, Optional, Union, Any

class StackedUnet(SegmentationModel):
    def __init__(
            self,
            encoder_depth: int = 4,
            encoder_name: str = "timm-efficientnet-b4",
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: any=None,
            num_layers: int=3,
            **kwargs: dict[str, Any]
        ):
        super().__init__()

        self.encoder = nn.ModuleList([
            get_encoder(
                encoder_name,
                in_channels=in_channels if i==0 else in_channels + 32,
                depth=encoder_depth,
                weights=encoder_weights if i==0 else None,
                **kwargs,
            )
            for i in range(num_layers)
        ]) 

        self.decoder = nn.ModuleList([
            UnetDecoder(
                encoder_channels=self.encoder[0].out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
            ) for i in range(num_layers)
        ])

        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=decoder_channels[-1], 
                out_channels=decoder_channels[-1] // 2,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1
            ),
            nn.BatchNorm2d(decoder_channels[-1] // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=decoder_channels[-1] // 2, 
                out_channels=decoder_channels[-1] // 4,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1
            ),
            nn.BatchNorm2d(decoder_channels[-1] // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                decoder_channels[-1] // 4, 
                out_channels=classes,
                kernel_size= 3, 
                padding=1
            )
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.num_layers = num_layers

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        for dec in self.decoder:
            initialize_decoder(dec)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder[0].output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        # Sanity check input shape
        self.check_input_shape(x)

        # Loop through the U-net N times    
        for l in range(self.num_layers):
            features = self.encoder[l](x) if l==0 else self.encoder[l](torch.cat([x, decoder_output], dim=1))
            decoder_output = self.decoder[l](*features)

        masks = self.segmentation_head(decoder_output)

        return masks
    
def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)