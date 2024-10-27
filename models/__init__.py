from . import configs
# from . import dexinet
# from . import dexiunet
import segmentation_models_pytorch as smp

def load_model(model_name: str="unet"):
    model_names_list = [
        "unet",
        "dexined-segmenter",
        "dexiunet"
    ]

    if model_name not in model_names_list:
        raise ModuleNotFoundError(f"No model, {model_name} exists.")
    
    else:
        match model_name:
            case "unet":
                model_config = configs.UnetConfig()
                model = smp.Unet(
                    encoder_name=model_config.encoder_name,
                    encoder_weights=model_config.encoder_weights,
                    in_channels=model_config.in_channels,
                    decoder_use_batchnorm=model_config.decoder_use_batchnorm,
                    decoder_attention_type=model_config.decoder_attention_type,
                    classes=model_config.classes
                )
            
            # case "dexined-segmenter":
            #     model_config = configs.DexinedSegmenterConfig()
            #     model = dexinet.DexinedSegmenter(
            #         classes=model_config.classes,
            #         activation=model_config.activation,
            #         pretrained=model_config.pretrained
            #     )

            # case "dexiunet":
            #     model_config = configs.DexiUnetConfig()
            #     model = dexiunet.DexiUnet(
            #         encoder_channels=model_config.encoder_channels,
            #         decoder_channels=model_config.decoder_channels,
            #         decoder_depth=model_config.decoder_depth,
            #         use_batchnorm=model_config.use_batchnorm,
            #         attention_type=model_config.attention_type,
            #         center=model_config.center,
            #         classes=model_config.classes, 
            #         activation=model_config.activation, 
            #         pretrained=model_config.pretrained
            #     )

        return model, model_config