import asyncio, os, argparse, glob
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
        slice_dim: str,
        q: CaseQueue, 
        volume_first: bool=False):
    
    assert slice_dim in ["first", "last"], f"slice_dim must be one of (first, last), got {slice_dim}"
    
    if volume_first:
        if slice_dim == "first":
            img_slice = nib.load(segmentation_path).get_fdata(caching="unchanged")[_slice]
        else:
            img_slice = nib.load(segmentation_path).get_fdata(caching="unchanged")[:, :, _slice]
    else:
        if slice_dim == "first":
            img_slice = nib.load(segmentation_path)[_slice]
        else:
            img_slice = nib.load(segmentation_path)[:, :, _slice]
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
        dataset_label: str,
        dataset_path: str, 
        q: CaseQueue, 
        volume_first: bool=True,
        n_concurrency: int=10):
    
    semaphore = asyncio.Semaphore(value=n_concurrency)
    tasks = []
    
    if dataset_label == "kits23":
        imaging_fname = "imaging.nii.gz"
        segmentation_fname = "segmentation.nii.gz"
        slice_dim = "first"

        case_paths = [os.path.join(dataset_path, case) for case in os.listdir(dataset_path)]

        print("scan cases...")
        for case_path in tqdm(case_paths):
            _case = case_path.split("/")[-1]
            imaging_path = os.path.join(case_path, imaging_fname)
            segmentation_path = os.path.join(case_path, segmentation_fname)
            nslices = nib.load(segmentation_path).dataobj.shape[0]

            for _slice in range(nslices):
                args = (_case, imaging_path, segmentation_path, _slice, slice_dim, volume_first)
                tasks.append(record_case_coroutine(args, q, semaphore))

    elif dataset_label == "lits23":
        slice_dim = "last"
        segmentation_paths = glob.glob(f"{dataset_path}*/segmentation-*.nii")
        segmentation_paths = sorted(segmentation_paths, key=lambda i : int(i.split("-")[-1].replace(".nii", "")))

        imaging_paths = glob.glob(f"{dataset_path}*/volume-*.nii")
        imaging_paths = sorted(imaging_paths, key=lambda i : int(i.split("-")[-1].replace(".nii", "")))

        assert len(imaging_paths) == len(segmentation_paths), "number of volumes do not match number of segments"

        for (imaging_path, segmentation_path) in tqdm(zip(imaging_paths, segmentation_paths)):
            _case = imaging_path.split("-")[-1].replace(".nii", "").zfill(5)
            _case = f"case_{_case}"
            nslices = nib.load(segmentation_path).dataobj.shape[-1]

            for _slice in range(nslices):
                args = (_case, imaging_path, segmentation_path, _slice, slice_dim, volume_first)
                tasks.append(record_case_coroutine(args, q, semaphore))

        print("scan cases...")

    print("\nscan slices...")
    return [await task for task in tqdm(asyncio.as_completed(tasks), total=len(tasks))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Generate MetaFile")
    parser.add_argument("--dataset", type=str, default="kits23", choices=["kits23", "lits17"], metavar="", help="dataset to generate metafile for")
    parser.add_argument("--volume_first", type=int, default=0, choices=[0, 1], metavar="", help="load entire volume and then slice")
    parser.add_argument("--n_concurrency", type=int, default=100, metavar="", help="number of concurrent coroutines to execute at a time")

    args = parser.parse_args()
    n_concurrency = args.n_concurrency
    volume_first = bool(args.volume_first)

    if args.dataset == "kits23":
        dataset_path = "kits23/dataset/"
        csv_path="kits23_metadata.csv"

    elif args.dataset == "lits17":
        dataset_path = "lits17/dataset/"
        csv_path="lits17_metadata.csv"


    if not os.path.isfile(csv_path):
        q = CaseQueue()
        asyncio.run(
            async_record(args.dataset, dataset_path, q, n_concurrency=n_concurrency, volume_first=volume_first)
        )
        q.to_df(to_csv=True, csv_path=csv_path)
    else:
        print(f"metafile already exists at: {csv_path}")