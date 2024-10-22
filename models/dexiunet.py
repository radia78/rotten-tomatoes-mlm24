from torch import nn
import torch
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from segmentation_models_pytorch.base import modules as md
from kornia.contrib.edge_detection import DexiNed

class DexiNedEncoder(DexiNed):
    def forward(self, x):
        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)  # [128,256,50,50]
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down + block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side

        # Block 5
        block_5_pre_dense = self.pre_dense_5(block_4_down)  # block_5_pre_dense_512 +block_4_down
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])

        # return results
        return [block_1, block_2, block_3, block_4, block_5]
    
class DexiUnet(nn.Module):
    def __init__(
            self, 
            encoder_channels,
            decoder_channels,
            decoder_depth,
            use_batchnorm,
            attention_type,
            center,
            classes, 
            activation, 
            pretrained
        ):
        super(DexiUnet, self).__init__()
        
        # Freeze the weights of the encoder
        self.encoder = DexiNedEncoder(pretrained=pretrained)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=decoder_depth,
            use_batchnorm=use_batchnorm,
            attention_type=attention_type,
            center=center
        )
        self.final_decoder_blocks = DecoderBlock(
            in_channels=encoder_channels[0],
            skip_channels=encoder_channels[0],
            out_channels=encoder_channels[0]//2,
            use_batchnorm=use_batchnorm,
            attention_type=attention_type
        )
        self.segmentation_head = SegmentationHead(
            in_channels=encoder_channels[0]//2,
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
        enc_output = self.encoder(x)
        dec_output = self.decoder(*enc_output)
        concat_dec_output = torch.concat((dec_output, enc_output[0]), dim=1)
        seg_input = self.final_decoder_blocks(concat_dec_output)

        # Pass through the segmentation head to generate masks
        masks = self.segmentation_head(seg_input)

        return masks
