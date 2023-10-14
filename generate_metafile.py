import asyncio, os, argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import nibabel as nib
from typing import Optional, Tuple

class CaseQueue:
    def __init__(self):
        self.img_paths = []
        self.segment_paths = []
        self.cases = []
        self.slices = []
        self.kidneys = []
        self.tumors = []
        self.cysts =  []

    def enqueue(
            self, 
            img_path: str, 
            segment_path: str, 
            case: str, 
            slice: int, 
            kidney: bool, 
            tumor: bool, 
            cyst: bool):
        
        self.img_paths.append(img_path)
        self.segment_paths.append(segment_path)
        self.cases.append(case)
        self.slices.append(f"slice_{str(slice).zfill(5)}")
        self.kidneys.append(kidney)
        self.tumors.append(tumor)
        self.cysts.append(cyst)

    def to_df(self, to_csv: bool=False, csv_path: Optional[str]=None) -> pd.DataFrame:
        df = pd.DataFrame()
        df["img_path"] = self.img_paths
        df["segment_path"] = self.segment_paths
        df["case"] = self.cases
        df["slice"] = self.slices
        df["kidney"] = self.kidneys
        df["tumor"] = self.tumors
        df["cyst"] = self.cysts

        if to_csv:
            if csv_path is not None:
                df.to_csv(csv_path, index=False)
        return df


def record_case(
        _case: str, 
        imaging_path: str, 
        segmentation_path: str,
        _slice: int, 
        q: CaseQueue, 
        volume_first: bool=False):
    
    if volume_first:
        img_slice = nib.load(segmentation_path).get_fdata(caching="unchanged")[_slice]
    else:
        img_slice = nib.load(segmentation_path)[_slice]
    kidney, tumor, cyst = False, False, False
    
    uids = np.unique(img_slice)
    if 1 in uids: kidney = True
    if 2 in uids: tumor = True
    if 3 in uids: cyst = True

    q.enqueue(imaging_path, segmentation_path, _case, _slice, kidney, tumor, cyst)


async def record_case_coroutine(args: Tuple, q: CaseQueue, semaphore: asyncio.Semaphore):
    loop = asyncio.get_running_loop()
    async with semaphore:
        await loop.run_in_executor(None, lambda: record_case(*args, q=q))


async def async_record(
        dataset_path: str, 
        imaging_fname: str, 
        segmentation_fname: str, 
        q: CaseQueue, 
        volume_first: bool=True,
        n_concurrency: int=10):
    
    case_paths = [os.path.join(dataset_path, case) for case in os.listdir(dataset_path)]
    semaphore = asyncio.Semaphore(value=n_concurrency)
    tasks = []

    print("scan cases...")
    for case_path in tqdm(case_paths):
        _case = case_path.split("/")[-1]
        imaging_path = os.path.join(case_path, imaging_fname)
        segmentation_path = os.path.join(case_path, segmentation_fname)
        nslices = nib.load(segmentation_path).dataobj.shape[0]

        for _slice in range(nslices):
            args = (_case, imaging_path, segmentation_path, _slice, volume_first)
            tasks.append(record_case_coroutine(args, q, semaphore))
    
    print("\nscan slices...")
    return [await task for task in tqdm(asyncio.as_completed(tasks), total=len(tasks))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Generate MetaFile")
    parser.add_argument("--volume_first", type=int, default=0, choices=[0, 1], metavar="", help="load entire volume and then slice")
    parser.add_argument("--n_concurrency", type=int, default=100, metavar="", help="number of concurrent coroutines to execute at a time")

    args = parser.parse_args()
    n_concurrency = args.n_concurrency
    volume_first = bool(args.volume_first)

    dataset_path = "kits23/dataset/"
    imaging_fname = "imaging.nii.gz"
    segmentation_fname = "segmentation.nii.gz"
    csv_path="kits23_metadata.csv"

    if not os.path.isfile(csv_path):
        q = CaseQueue()
        asyncio.run(
            async_record(
                dataset_path, 
                imaging_fname, 
                segmentation_fname, 
                q, 
                n_concurrency=n_concurrency, 
                volume_first=volume_first
            )
        )
        q.to_df(to_csv=True, csv_path=csv_path)

    else:
        print(f"metafile already exists at: {csv_path}")