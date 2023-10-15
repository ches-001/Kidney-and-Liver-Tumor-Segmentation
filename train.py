import torch, os, gc, random, argparse
import torch.nn as nn
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import Dict, Any, Tuple, Optional, Callable, List

# custom modules
from data_utils import CTImageDataset
from metrics import BinaryFocalLoss, HybridBinaryEntropyLoss
from pipelines import TrainingPipeline
from seg_network import SegmentationNet
from transforms import data_augmenter

os.environ["SEED"] = "3047"
os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
nib.openers.HAVE_INDEXED_GZIP = True
nib.arrayproxy.KEEP_FILE_OPEN_DEFAULT = True

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
SEED = int(os.getenv("SEED"))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

METAFILE_PATH = "kits23_metadata.csv"


def create_dataset(
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        *,
        img_size: Tuple[int, int]=(256, 256), 
        cache_size: int=55_000, 
        train_data_size: float=0.9,
        transforms_kwargs: Dict[str, Any]={"add_noise": False},
        dataset_transform_kwargs: Dict[str, Any]={"p": 0.4},
        volume_first: bool=False,
    ) -> Tuple[CTImageDataset, CTImageDataset]:

    data_transforms = data_augmenter(**transforms_kwargs)

    train_cache_size = (cache_size * train_data_size)
    eval_cache_size = cache_size - train_cache_size

    train_dataset = CTImageDataset(
        train_df, 
        data_transforms, 
        dataset_transform_kwargs,
        img_size=img_size, 
        cache_size=train_cache_size,
        binary_segment=True,
        volume_first=volume_first,
    )

    eval_dataset = CTImageDataset(
        eval_df, 
        img_size=img_size, 
        cache_size=eval_cache_size,
        binary_segment=True,
        volume_first=volume_first,
    )

    return train_dataset, eval_dataset


def train(
        train_dataset: CTImageDataset, 
        eval_dataset: CTImageDataset,
        *,
        batch_size: int=32,
        shuffle: bool=True,
        num_workers: int=os.cpu_count(),
        loss_func_id: str="bce",
        bce_focal_weights: Tuple[float, float]=(0.3, 0.7),
        with_logits: bool=False,
        label_smoothing: Optional[float]=None,
        loss_reduction: str="mean",
        alpha: float=0.25,
        gamma: float=2.0,
        input_channels: int=1,
        last_fmap_channels: int=512, #2048
        num_classes: int=3,
        block: Callable=BasicBlock,
        block_layers: List=[3, 4, 6, 3],
        T_0: int=20,
        epochs: int=140,
        lr: float=1e-3,
        weight_decay: float=0.0,
        min_lr: float=1e-6,
        T_mult: int=2,
        xavier_weight_init: bool=True,
        log_train_step: int=1,
        verbose: bool=True,
        plot_metrics: bool=True,
        save_metrics: bool=True,
    ):

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )

    if loss_func_id == "hybrid":
        lossfunc = HybridBinaryEntropyLoss(
            bce_focal_weights=bce_focal_weights,
            with_logits=with_logits,
            label_smoothing=label_smoothing,
            reduction=loss_reduction,
            focal_loss_kwargs={"alpha": alpha, "gamma": gamma}
        )
    elif loss_func_id == "focal":
        lossfunc = BinaryFocalLoss(
            alpha=alpha, 
            gamma=gamma, 
            with_logits=with_logits, 
            label_smoothing=label_smoothing, 
            reduction=loss_reduction
        )
    elif loss_func_id == "bce":
        if not with_logits:
            lossfunc = nn.BCELoss(reduction=loss_reduction)
        else:
            lossfunc = nn.BCEWithLogitsLoss(reduction=loss_reduction)
    else:
        raise ValueError("loss_func_id can only be one of 'hybrid', 'focal', 'bce'")

    model = SegmentationNet(
        input_channels, 
        num_classes=num_classes,
        block=block,
        block_layers=block_layers,
        last_fmap_ch=last_fmap_channels,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pipeline = TrainingPipeline(
        model=model, 
        optimizer=optimizer, 
        device=os.getenv("DEVICE"), 
        lossfunc=lossfunc,
        xavier_weight_init=xavier_weight_init,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        pipeline.optimizer, 
        T_0=T_0, 
        T_mult=T_mult, 
        eta_min=min_lr,
        verbose=verbose
    )

    best_loss = np.inf
    
    for epoch in range(epochs):  
        if (epoch) % log_train_step == 0:
            train_verbosity = verbose
        else:
            train_verbosity = False
        pipeline.train(train_dataloader, verbose=train_verbosity)
        eval_loss, *_ = pipeline.evaluate(eval_dataloader, verbose=train_verbosity)

        if eval_loss < best_loss: 
            best_loss = eval_loss
            pipeline.save_model()
            print(f"model saved at epoch: {epoch}")

        lr_scheduler.step()

    if plot_metrics:
        pipeline.plot_metrics()

    if save_metrics:
        pipeline.save_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Train Segmentation Network")
    parser.add_argument("--total_samples", type=int, default=26_000, metavar="", help="total number of samples to split as train and test. (Values beyond 50_000 will be clamped to 50_000)")
    parser.add_argument("--train_size", type=float, default=0.9, metavar="", help="% train data size")
    parser.add_argument("--cache_size", type=int, default=55_000, metavar="", help="Number of sample pairs in cache memory")
    parser.add_argument("--img_size", type=int, default=256, metavar="", help="size of image data for (H and W)")
    parser.add_argument("--volume_first", type=int, default=0, choices=[0, 1], metavar="", help="load entire volume and then slice")


    args = parser.parse_args()

    meta_df = pd.read_csv(METAFILE_PATH)

    plain_img_df = meta_df[meta_df["kidney"] != True]
    plain_img_df = plain_img_df[plain_img_df["tumor"] != True]
    plain_img_df = plain_img_df[plain_img_df["cyst"] != True]
    segment_img_df = meta_df[~meta_df.isin(plain_img_df)].dropna()

    plain_img_df.index = range(0, len(plain_img_df))
    segment_img_df.index = range(0, len(segment_img_df))
    data_df = pd.concat([segment_img_df, plain_img_df.sample(frac=15_000/plain_img_df.shape[0])], axis=0)

    data_df = data_df.sample(n=np.clip(args.total_samples, a_min=0, a_max=50_000))
    data_df.index = range(0, data_df.shape[0])
    del meta_df, plain_img_df, segment_img_df
    gc.collect()

    train_df = data_df.sample(frac=args.train_size)
    eval_df = data_df[~data_df.isin(train_df)].dropna()
    train_df.index = range(0, train_df.shape[0])
    eval_df.index = range(0, eval_df.shape[0])

    train_dataset, eval_dataset = create_dataset(
        train_df, 
        eval_df, 
        img_size=(args.img_size, args.img_size), 
        cache_size=args.cache_size, 
        train_data_size=args.train_size, 
        volume_first=bool(args.volume_first)
    )

    train(train_dataset, eval_dataset)