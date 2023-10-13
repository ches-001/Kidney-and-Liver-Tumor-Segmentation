import torch, random
from typing import Union, Optional

class Noise: 
    
    @staticmethod
    def apply_gaussian(
        image: torch.Tensor, mean: Optional[Union[int, float]]=0, std: Optional[Union[int, float]]=1): 
        noise = torch.normal(mean=mean, std=std, size=image.shape)
        image = image + noise
        return image

    @staticmethod
    def apply_speckle(
        image: torch.Tensor, mean: Optional[Union[int, float]]=0, std: Optional[Union[int, float]]=1): 
        speckle = image * Noise.apply_gaussian(image, mean, std)
        image = image + speckle
        return image

    @staticmethod
    def apply_salt_and_pepper(image: torch.Tensor, n_pixels: int=500): 
        _, H, W = image.shape

        for op in ['salt', 'pepper']: 
            val = image.max()
            if op == 'pepper': val = 0
            
            for i in range(n_pixels): 
                x_coord = random.randint(0, W-1)
                y_coord = random.randint(0, H-1)
                image[: , y_coord, x_coord] = val
        return image
        
    @staticmethod
    def apply_poisson(image: torch.Tensor, rate: Optional[Union[int, float]]=2): 
        noise = torch.rand(*image.shape) * rate
        noise = torch.poisson(noise)
        image = image + noise
        return image