"""ImageNet geometry used by the frozen classifier and GEAR views."""
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def transform_for(crop=224, resize=256):
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
