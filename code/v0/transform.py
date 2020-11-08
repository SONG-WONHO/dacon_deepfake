from albumentations.pytorch import ToTensor
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Normalize, Cutout, PadIfNeeded,
    RandomCrop, ToFloat, RandomGridShuffle, ChannelShuffle, GridDropout,
    OneOf, RandomRotate90, RandomResizedCrop, Resize
)


def transform_v0(config):
    """ default transforms

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Resize(config.image_size, config.image_size),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    test_transforms = Compose([
        Resize(config.image_size, config.image_size),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    return train_transforms, test_transforms


def get_transform(config):
    try:
        name = f"transform_v{config.transform_version}"
        f = globals().get(name)
        print(f"... Transform Info - {name}")
        return f(config)

    except TypeError:
        raise NotImplementedError("try another transform version ...")