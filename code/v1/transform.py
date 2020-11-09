import cv2
from albumentations.pytorch import ToTensor
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Normalize, Cutout, PadIfNeeded,
    RandomCrop, ToFloat, RandomGridShuffle, ChannelShuffle, GridDropout,
    OneOf, RandomRotate90, RandomResizedCrop, Resize, ImageCompression,
    GaussNoise, GaussianBlur, RandomBrightnessContrast, FancyPCA,
    HueSaturationValue, ToGray, ShiftScaleRotate
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


def transform_v1(config):
    train_transforms = Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        Resize(config.image_size, config.image_size),
        PadIfNeeded(min_height=config.image_size,
                    min_width=config.image_size,
                    border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(),
               FancyPCA(),
               HueSaturationValue()], p=0.7),
        ShiftScaleRotate(shift_limit=0.1,
                         scale_limit=0.2,
                         rotate_limit=10,
                         border_mode=cv2.BORDER_CONSTANT,
                         p=0.5),
    ])

    test_transforms = Compose([
        Resize(config.image_size, config.image_size),
        PadIfNeeded(min_height=config.image_size,
                    min_width=config.image_size,
                    border_mode=cv2.BORDER_CONSTANT),
    ])

    return train_transforms, test_transforms


def get_transform(config):
    try:
        name = f"transform_v{config.transform_version}"
        f = globals().get(name)
        print(f"... Transform Info - {name}")
        return f(config)

    except TypeError:
        raise NotImplementedError("try another transform version ...")