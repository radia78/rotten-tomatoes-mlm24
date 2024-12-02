import kornia
import torch
import torch.nn as nn
from torchvision.ops import roi_align
from segmentation_models_pytorch import Unet

class PointRefineUnet(Unet):
    def __init__(
            self, 
            encoder_name = "resnet34", 
            encoder_depth = 5, 
            encoder_weights = "imagenet", 
            decoder_use_batchnorm = True, 
            decoder_channels = [256, 128, 64, 32, 16], 
            decoder_attention_type = None, 
            in_channels = 3, 
            classes = 1, 
            activation = None, 
            aux_params = None
        ):
        # Initalize the constructor with the U-Net
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
            aux_params
        )

        # Initialize the small MLP for refinement
        self.mlp = torch.nn.Sequential(
            nn.Linear(self.encoder.out_channels[-1], decoder_channels[0]),
            nn.ReLU(),
            nn.Linear(decoder_channels[0], decoder_channels[1]),
            nn.Linear(decoder_channels[2], classes)
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # Standard forward method
        if not torch.jit.is_tracing():
            self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        coarse_masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return coarse_masks, labels

        # Point refinement methohd
        N, C, H, W = x.shape
        spatial_scale = features[-1].shape[-1] / W

        # Shape (N * k, feature_dim, 1, 1)
        indices, point_features = self.get_aligned_features(features[-1], coarse_masks, W, spatial_scale, num_points=1024)
        point_features = point_features.flatten(start_dim=1)
        # Shape (N * k, 1)
        refined_points = self.mlp(point_features)

        return indices, coarse_masks, refined_points

    def low_confidence_mask(self, coarse_mask, tau=(0.7, 0.1)):
        A = torch.where((coarse_mask.sigmoid() < tau[0]) & (coarse_mask.sigmoid() > tau[-1]), 1.0, 0.0)
        return torch.abs(A * coarse_mask.sigmoid() - 0.5)

    def sample_uncertain_points(self, confidence_mask, num_points):
        batch = confidence_mask.shape[0]
        flat_uncertainity = confidence_mask.view(batch, -1) # Flattens dimensions except batch
        indices = torch.topk(flat_uncertainity, k=num_points, dim=1, largest=False).indices
        return indices
    
    def get_bounding_boxes(self, batch_indices, mask_width):
       device = batch_indices.device
       batches = torch.arange(len(batch_indices), device=device).unsqueeze(1).expand(-1, batch_indices.size(1))  # Shape: (n, k)
       x_coords = batch_indices % mask_width
       y_coords = batch_indices // mask_width
       rois = torch.stack([batches, x_coords, y_coords, x_coords + 1, y_coords + 1], dim=2).view(-1, 5).float()
       
       return rois
    
    def get_aligned_features(self, features, coarse_mask, mask_width, spatial_scale, num_points):
        unconfidence_mask = self.low_confidence_mask(coarse_mask)
        indices = self.sample_uncertain_points(unconfidence_mask, num_points=num_points)
        rois = self.get_bounding_boxes(indices, mask_width)
        return indices, roi_align(features, rois, output_size=(1, 1), spatial_scale=spatial_scale, sampling_ratio=2, aligned=True)