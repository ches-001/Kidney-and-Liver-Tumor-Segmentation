import torch, random
from torchvision import transforms
from ._noise import Noise


class FirstChannelRandomGaussianBlur(object): 
    def __init__(self, p: float, kernel_size: tuple=(3, 9), sigma: tuple=(0.1, 1)): 
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)

    def __call__(self, sample: torch.Tensor): 
        #input shape:  (C=2, H, W)
        if self.p < random.random(): return sample

        image, mask = sample[0].unsqueeze(dim=0), sample[1: ]
        image = self.gaussian_blur(image)
        sample = torch.cat((image, mask), dim=0)
        return sample


class FirstChannelRandomNoise(object): 
    def __init__(self, p: float=0.5, **kwargs): 
        defaultKwargs = {
            'mean_range': (0, 1),
            'std_range': (1, 3),
            'sp_px_range': (2, 500),
            'poisson_rate_range': (1, 3)}
        
        self.p = p
        self.kwargs = {**defaultKwargs, **kwargs}
        self.ops = ('gaussian', 'speckle', 'salt_&_pepper', 'poisson')

    def __call__(self, sample: torch.Tensor): 
        #input shape:  (C=2, H, W)
        if self.p < random.random():  return sample

        image, mask = sample[0].unsqueeze(dim=0), sample[1: ]
        op = random.choice(self.ops)
        if op == 'gaussian': 
            mean = random.randint(*self.kwargs['mean_range'])
            std = random.randint(*self.kwargs['std_range'])
            image = Noise.apply_gaussian(image, mean, std)

        elif op == 'speckle': 
            mean = random.randint(*self.kwargs['mean_range'])
            std = random.randint(*self.kwargs['std_range'])
            image = Noise.apply_speckle(image, mean, std)

        elif op == 'salt_&_pepper': 
            n_pixels = random.randint(*self.kwargs['sp_px_range'])
            image = Noise.apply_salt_and_pepper(image, n_pixels)

        elif op == 'poisson': 
            rate = random.randint(*self.kwargs['poisson_rate_range'])
            image = Noise.apply_poisson(image, rate)

        sample = torch.cat((image, mask), dim=0)
        return sample

        
class CustomRandomRotation(object): 
    def __init__(self, p: float, angle_range: tuple=(0, 360)): 
        self.p = p
        self.angle_range = angle_range
        self.random_rotation = transforms.RandomRotation(self.angle_range)

    def __call__(self, sample: torch.Tensor): 
        #input shape:  (C=1, H, W)
        if self.p < random.random(): return sample

        sample = self.random_rotation(sample)
        return sample

  
class CustomRandomResizedCrop(object): 
    def __init__(self, p: float, scale: tuple=(0.5, 1.0)): 
        self.p = p
        self.scale = scale

    def __call__(self, sample: torch.Tensor): 
        #input shape:  (C=1, H, W)
        if self.p < random.random(): return sample

        _, H, W = sample.shape
        random_resizedcrop = transforms.RandomResizedCrop(
            (H, W), scale=self.scale, interpolation=transforms.InterpolationMode.NEAREST)
        sample = random_resizedcrop(sample)
        return sample


class CustomCompose(transforms.Compose): 
    def __init__(self, transforms: list, shuffle: bool=False): 
        super(CustomCompose, self).__init__(transforms)
        self.transforms = transforms
        self.shuffle = shuffle

    def __call__(self, image: torch.Tensor): 
        transforms = self.transforms
        if self.shuffle: 
            transforms = random.sample(self.transforms, len(self.transforms))
            
        for T in transforms: 
            image = T(image)
        return image