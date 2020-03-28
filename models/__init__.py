from models.deeplab import DeepLab3D
from models.unet import Unet3D
from models.attention_gated_unet import AttentionGatedUnet3D


def get_model(config):
    if config.name == 'unet':
        return Unet3D(config)
    if config.name == "attention_unet":
        return AttentionGatedUnet3D(config)
    if config.name == 'deeplab':
        return DeepLab3D(config)
