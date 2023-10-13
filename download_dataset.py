import asyncio, tqdm, os, argparse
from urllib import error
from kits23 import TRAINING_CASE_NUMBERS
from kits23.download import download_dataset, download_case, get_destination, DST_PTH


async def download_case_coroutine(
        case_num: int, 
        pbar: tqdm.tqdm, 
        semaphore: asyncio.Semaphore, 
        retry: bool=True):
    
    loop = asyncio.get_running_loop()
    async with semaphore:
        await loop.run_in_executor(None, lambda: download_case(case_num, pbar, retry))

    
async def async_download_dataset(n_concurrency: int=10):
    DST_PTH.mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(value=n_concurrency)
    tasks = []

    # Determine which cases still need to be downloaded
    left_to_download = []
    for case_num in TRAINING_CASE_NUMBERS:
        case_id = f"case_{case_num:05d}"
        dst = get_destination(case_id)
        if not dst.exists():
            left_to_download = left_to_download + [case_num]

    print(f"\nFound {len(left_to_download)} cases to download\n")

    for case_num in (pbar := tqdm.tqdm(left_to_download)):
        pbar.set_description(f"Adding case_{case_num:05d} coroutine to futures task")
        tasks.append(download_case_coroutine(case_num, pbar, semaphore=semaphore, retry=True))
    
    return [await _ for _ in tqdm.tqdm(asyncio.as_completed(tasks), total=len(left_to_download))]            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Dataset Downloader")
    parser.add_argument("--async_download", type=int, default=int(True), choices=[0, 1], metavar="", help="download dataset asynchronously")
    parser.add_argument("--n_concurrency", type=int, default=10, metavar="", help="number of concurrent coroutines to execute at a time")

    args = parser.parse_args()
    n_concurrency = args.n_concurrency
    async_download = bool(args.async_download)
    
    dataset_dir = "kits23/dataset/"
    imaging_name = "imaging.nii.gz"
    cases = [os.path.join(dataset_dir, case_path) for case_path in os.listdir(dataset_dir)]
    done = False


    while not done:
        try:
            if async_download:
                asyncio.run(async_download_dataset(n_concurrency))
            else:
                download_dataset()
        except error.ContentTooShortError as e:
            print(e)

        if all([imaging_name in os.listdir(case) for case in cases]):
            done = True