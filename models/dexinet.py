from torch import nn
import torch
from segmentation_models_pytorch.base import SegmentationHead
from kornia.contrib.edge_detection import DexiNed

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.amax(x, dim=1).unsqueeze(1), torch.mean(x, dim=1).unsqueeze(1)), dim=1
        )
    
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class DexinedSegmenter(nn.Module):
    def __init__(self, classes=1, activation=None, pretrained=True):
        super(DexinedSegmenter, self).__init__()
        
        # Freeze the weights of the encoder
        self.encoder = DexiNed(pretrained=pretrained)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.spatial_attention = SpatialGate()
        self.segmentation_head = SegmentationHead(
            in_channels=6,
            out_channels=classes,
            activation=activation,
            kernel_size=3
        )

        self.apply(self._initialize_trainable_weights)

    def _initialize_trainable_weights(self, m):
        if isinstance(m, (nn.Conv2d, )):
            nn.init.xavier_normal_(m.weight, gain=1.0)
        
            if m.weight.data.shape[1] == torch.Size([1]):
                nn.init.normal_(m.weight, mean=0.0)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Pass the image through the Dexined encoder
        features = self.encoder(x)
        fused_features = torch.concat(features[:-1], dim=1) * -1
        attn_output = self.spatial_attention(fused_features)

        # Pass through the segmentation head to generate masks
        masks = self.segmentation_head(attn_output)

        return masks
