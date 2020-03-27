from models.deeplab import DeepLab3D
from models.unet import Unet3D


def get_model(config, loss_type=None):
    if config.name == 'unet':
        return Unet3D(config, loss_type)
    if config.name == 'deeplab':
        return DeepLab3D(config, loss_type)
