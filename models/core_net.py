import torch
from torch import nn
from kornia.filters import sobel, spatial_gradient
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from typing import List, Optional

class CoreNet(nn.Module):
    def __init__(
            self,
            encoder_name: str,
            in_channels: int,
            encoder_weights: Optional[str],
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            encoder_depth: int=5,
            use_batchnorm: bool=True,
            decoder_attention_type: Optional[str]=None
        ):
        super(CoreNet).__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=use_batchnorm,
            attention_type=decoder_attention_type
        )

        self.dac = DACBlock(channels=self.encoder.out_channels)
        self.rmp = RMPBlock(channels=self.encoder.out_channels)

        self.point_head = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    def confidence_mask_module(
            self,
            mask_p: torch.Tensor, 
            confidence_range: List[float]=[0.9, 0.1]
        ):

        A = torch.where((mask_p >= confidence_range[1]) & (mask_p <= confidence_range[0]), 1, 0)
        confidence_mask = torch.abs((1 - A) * mask_p - 0.5)

        return confidence_mask
    
    def breaking_points_module(self, mask_p: torch.Tensor, tau: int=4):
        G = sobel(mask_p) # Compute the magnitude of the sobel operator
        G_A = torch.where((G > 0), 1, 0)
        K = K = torch.tensor(
            [1, 1, 1],
            [1, -10, 1],
            [1, 1, 1,]
        ).unequeeze(0).unsqueeze(0)
        G_A_K = nn.functional.conv2d(
            input=G_A,
            weight=K,
            stride=1,
            padding=0
        )
        
        return torch.where((G_A_K > tau), 1, 0)
    
    def branching_points_module(mask_p: torch.Tensor):
        # Detect the branching points of the mask
        G_spatial = spatial_gradient(mask_p)
        G_x, G_y = G_spatial[:, :, 0, :, :], G_spatial[:, :, 1, :, :]

        div_x = torch.diff(G_x, dim=-1)
        div_y = torch.diff(G_y, dim=-2)

        divergence = div_x + div_y

        return divergence

class DACBlock(nn.Module):
    def __init__(self, channels: int):
        super(DACBlock).__init__()

        self.cascade_blocks_list = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1
        )

        self.cascade_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=3
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1
            )
        )

        self.cascade_3 = nn.Sequential(
            self.cascade_1,
            self.cascade_2
        )

        self.cascade_4 = nn.Sequential(
            self.cascade_3[:-1],
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=5
            ),
            self.cascade_3[-1]
        )

    def forward(self, x):
        csc_out = self.cascade_1(x)
        csc_out += self.cascade_2(x)
        csc_out += self.cascade_3(x)
        csc_out += self.cascade_4(x)

        return csc_out

class RMPBlock(nn.Module):
    def __init__(
            self, 
            channels: int,
            kernel_sizes: List[int]=[2, 3, 5, 6]
        ):
        super(RMPBlock).__init__()

        self.max_pools = nn.ModuleList([
            nn.MaxPool2d(
                in_channels=channels,
                kernel_size=k, 
                stride=1
            ) for k in kernel_sizes
        ])

        self.conv1x1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        _, _, H, W = x.shape # N, C, H, W
        pool_out = []
        for pool_layer in self.pools:
            out = self.conv1x1(pool_layer(x))
            pool_out.append(nn.functional.upsample_bilinear(out, size=(H, W)))
        
        output = torch.concat(x + pool_out, dim=1) # Concat on the channel dimension

        return output          
def breaking_points_module(
    coarse_mask_p: torch.Tensor,
    k: int=4
    ):

    G = sobel(coarse_mask_p)
    G_A = torch.where((G > 0), 1, 0)
    K = K = torch.tensor(
        [1, 1, 1],
        [1, -10, 1],
        [1, 1, 1,]
    ).unequeeze(0).unsqueeze(0)

    return nn.functional.conv2d(G_A, K, stride=1, padding=0)