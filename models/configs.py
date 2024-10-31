from dataclasses import dataclass 

"""
File contains all the configurations for the model
"""

@dataclass
class UnetConfig:
    encoder_name: str="resnet34"
    encoder_weights: str="imagenet"
    in_channels: int=3
    decoder_use_batchnorm: bool=True
    decoder_attention_type: str="scse"
    classes: int=1

@dataclass
class UnetConfig_NoPretrained:
    encoder_name: str="resnet34"
    encoder_weights: str=None
    in_channels: int=3
    decoder_use_batchnorm: bool=True
    decoder_attention_type: str="scse"
    classes: int=1

@dataclass
class DexinedSegmenterConfig:
    classes: int=1
    activation: any=None
    pretrained: bool=True

@dataclass
class DexiUnetConfig:
    encoder_channels: any=(64, 128, 256, 512, 512)
    decoder_channels: any=(256, 128, 64)
    decoder_depth: int=3
    use_batchnorm: bool=True
    attention_type: str="scse"
    center: bool=False
    classes: int=1
    activation: any=None 
    pretrained: bool=True
