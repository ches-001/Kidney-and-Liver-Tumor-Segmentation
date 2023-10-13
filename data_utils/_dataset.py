import torch
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Manager
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Any, Tuple
from transforms import img_normalize, img_resize


class CTImageDataset(Dataset):
    def __init__(
            self, 
            meta_df: pd.DataFrame, 
            data_transform: Optional[Callable]=None, 
            data_transform_kwargs: Dict[str, Any]={"p": 0.5},
            img_size: Tuple=(256, 256),
            cache_size: int=1_000,
            binary_segment: bool=True,
            volume_first: bool=False):
        
        self.meta_df = meta_df
        self.data_transform = data_transform
        self.data_transform_kwargs = data_transform_kwargs
        self.img_size = img_size
        self.cache_size = cache_size
        self.binary_segment = binary_segment
        self.volume_first = volume_first
        self.cache_memory = Manager().dict()

    def __len__(self) -> int:
        return self.meta_df.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if idx in self.cache_memory.keys():
            # retrieve from cache
            img, segment = self.cache_memory[idx]
            
        else:
            sample_df = self.meta_df.iloc[idx]
            img_path = sample_df["segment_path"]
            segment_path = sample_df["segment_path"]
            slice_idx = int(sample_df["slice"].split("_")[-1])

            if self.volume_first:
                # loading the entire volume from disk and indexing the slice 
                # is faster but memory inefficient
                img = nib.load(img_path).get_fdata(caching="unchanged")[slice_idx]
                segment = nib.load(segment_path).get_fdata(caching="unchanged")[slice_idx]
            else:
                # loading only slice from disk is slow but memory efficient
                img = nib.load(img_path).dataobj[slice_idx]
                segment = nib.load(segment_path).dataobj[slice_idx]

            if not img.flags["WRITEABLE"]:
                img = img.copy()

            if not segment.flags["WRITEABLE"]:
                segment = segment.copy()
            
            # convert from <f8> (float64) to int16 before caching to save memory
            img = img.astype(np.int16)

            # add to cache
            if self.cache_size > 0:
                if len(self.cache_memory) < self.cache_size:
                    self.cache_memory[idx] = (img, segment)

                else:
                    cache_keys = list(self.cache_memory.keys())
                    del self.cache_memory[cache_keys[0]]

        img = torch.from_numpy(img).unsqueeze(dim=0)
        segment = torch.from_numpy(segment).unsqueeze(dim=0)
        img = img_normalize(img)
        img = img_resize(img, size=self.img_size)
        segment = img_resize(segment, size=self.img_size)

        img_and_segment = torch.cat([img, segment], dim=0)

        if self.data_transform is not None:
            if np.random.uniform() < self.data_transform_kwargs["p"]:
                img_and_segment = self.data_transform(img_and_segment)
        
        img, segment = img_and_segment
        img = img.unsqueeze(dim=0).to(dtype=torch.float32)
        segment = segment.unsqueeze(dim=0).to(dtype=torch.float32)

        if self.binary_segment:
            kidney_label = 1; tumor_label = 2; cyst_label = 3
            binary_segment = torch.zeros((3, *self.img_size), dtype=segment.dtype)

            for i, label in enumerate([kidney_label, tumor_label, cyst_label]):
                template = segment.clone()
                template[template != label] = 0
                template[template == label] = 1
                binary_segment[i] = template
                
            segment = binary_segment

        return img, segment