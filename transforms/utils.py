import numpy as np
from torchvision import transforms
from typing import Union
from ._custom_transforms import *

def img_resize(
        img: Union[torch.Tensor, np.ndarray], 
        size: Union[int, tuple], 
        interpolation: str=transforms.InterpolationMode.BILINEAR) -> torch.Tensor: 
    
    #img shape:  C, H, W or N, C, H, W
    if isinstance(size, int):  H, W = size, size
    elif isinstance(size, tuple):  H, W = size
    else:  raise TypeError(f'size is required to be of type int fo tuple, {type(size)} given')

    assert len(img.shape) == 3 or len(img.shape) == 4, \
     'input img must be of shape  C, H, W or N, C, H, W'

    if isinstance(img, np.ndarray):  img = torch.from_numpy(img) 
    res_model = transforms.Resize((H, W), interpolation=interpolation, antialias=True)
    img = res_model(img)
    return img


def img_normalize(img: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]: 
    #img size:  (C, H, W)
    img = (img - img.min()) / (img.max() - img.min())
    return img    


def data_augmenter(add_noise: bool=False, **kwargs) -> CustomCompose: 

    defaultKwargs = {
        'shuffle_tranforms': True,
        'crop_p': 0.5, 
        'rotation_p': 0.5, 
        'Hflip_p': 0.5, 
        'Vflip_p': 0.5, 
        'blur_p': 0.5,
        'noise_p': 0.5,
        'rotation_angle_range': (-30, 30),
        'crop_scale': (0.7, 1.0),
        'blur_kernel_size': (3, 7),
        'blur_sigma': (0.1, 1.0),
        }

    kwargs = {**defaultKwargs, **kwargs}

    transform_list = [
        CustomRandomResizedCrop(p=kwargs['crop_p'], scale=kwargs['crop_scale']),
        CustomRandomRotation(p=kwargs['rotation_p'], angle_range=kwargs['rotation_angle_range']),
        transforms.RandomHorizontalFlip(kwargs['Hflip_p']),
        transforms.RandomVerticalFlip(kwargs['Vflip_p']),
        FirstChannelRandomGaussianBlur(kwargs['blur_p'], kernel_size=kwargs['blur_kernel_size'], sigma=kwargs['blur_sigma']),
    ]

    if add_noise:
        transform_list.append(FirstChannelRandomNoise(p=kwargs['noise_p']))

    T = CustomCompose(transform_list, shuffle=kwargs['shuffle_tranforms'])
    return T

if __name__ == '__main__': 
    data_transforms = data_augmenter()