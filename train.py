import torch, os, gc, random, argparse
import torch.nn as nn
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision.models import resnet
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

def create_dataset(
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        *,
        dataset_label: str="kits23",
        img_size: Tuple[int, int]=(224, 224),
        cache_size: int=55_000, 
        train_data_size: float=0.9,
        transforms_kwargs: Dict[str, Any]={"add_noise": False},
        dataset_transform_kwargs: Dict[str, Any]={"p": 0.4},
        volume_first: bool=False,
    ) -> Tuple[CTImageDataset, CTImageDataset]:

    data_transforms = data_augmenter(**transforms_kwargs)

    train_cache_size = (cache_size * train_data_size)
    eval_cache_size = cache_size - train_cache_size

    slice_dim = "first" if dataset_label=="kits23" else "last"

    if dataset_label == "kits23":
        label_ids = {"organ": 1, "tumor": 2, "cyst": 3}
        
    elif dataset_label == "lits17":
        label_ids = {"tumor": 1, "organ": 2}

    train_dataset = CTImageDataset(
        train_df, 
        data_transforms, 
        dataset_transform_kwargs,
        img_size=img_size, 
        cache_size=train_cache_size,
        binary_segment=True,
        volume_first=volume_first,
        slice_dim=slice_dim,
        label_ids=label_ids,
    )

    eval_dataset = CTImageDataset(
        eval_df, 
        img_size=img_size, 
        cache_size=eval_cache_size,
        binary_segment=True,
        volume_first=volume_first,
        slice_dim=slice_dim,
        label_ids=label_ids,
    )

    return train_dataset, eval_dataset


def train(
        train_dataset: CTImageDataset, 
        eval_dataset: CTImageDataset,
        *,
        dataset_label: Optional[str]=None,
        batch_size: int=32,
        shuffle: bool=True,
        num_workers: int=os.cpu_count(),
        loss_func_id: str="bce",
        bce_focal_weights: Tuple[float, float]=(0.5, 0.5),
        label_smoothing: Optional[float]=None,
        loss_reduction: str="mean",
        alpha: float=0.25,
        gamma: float=2.0,
        input_channels: int=1,
        last_fmap_channels: int=512, #2048
        num_classes: int=3,
        block: Callable=resnet.BasicBlock,
        block_layers: List=[3, 4, 6, 3],
        decay_lr: bool=True,
        T_0: int=20,
        epochs: int=140,
        lr: float=2e-4,
        weight_decay: float=0.0,
        min_lr: float=1e-6,
        T_mult: int=2,
        xavier_weight_init: bool=True,
        combine_metrics: bool=True,
        w_hd: float=0.5,
        w_dc: float=0.5,
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
            with_logits=False,
            label_smoothing=label_smoothing,
            reduction=loss_reduction,
            focal_loss_kwargs={"alpha": alpha, "gamma": gamma}
        )
    elif loss_func_id == "focal":
        lossfunc = BinaryFocalLoss(
            alpha=alpha, 
            gamma=gamma, 
            with_logits=False, 
            label_smoothing=label_smoothing, 
            reduction=loss_reduction
        )
    elif loss_func_id == "bce":
        lossfunc = nn.BCELoss(reduction=loss_reduction)
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
        combine_metrics=combine_metrics,
        w_hd=w_hd,
        w_dc=w_dc,
        model=model, 
        optimizer=optimizer, 
        device=os.getenv("DEVICE"), 
        lossfunc=lossfunc,
        xavier_weight_init=xavier_weight_init,
        dataset_label=dataset_label,
    )

    if decay_lr:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            pipeline.optimizer, 
            T_0=T_0, 
            T_mult=T_mult, 
            eta_min=min_lr,
            verbose=verbose
        )

    best_score = -np.inf

    if os.getenv("DEVICE") != "cpu":
        print(f"Model is training on {torch.cuda.get_device_name()} \n")
    
    for epoch in range(epochs):  
        if (epoch) % log_train_step == 0:
            train_verbosity = verbose
        else:
            train_verbosity = False
        pipeline.train(train_dataloader, verbose=train_verbosity)
        _, eval_dice_score, eval_hausdorff = pipeline.evaluate(eval_dataloader, verbose=train_verbosity)

        # compute model saving metric
        eval_score = (w_hd * (1 - eval_hausdorff)) + (w_dc * eval_dice_score)

        if eval_score > best_score: 
            best_score = eval_score
            pipeline.save_model()
            print(f"model saved at epoch: {epoch} with score value: {eval_score}")

        if decay_lr:
            lr_scheduler.step()

    if plot_metrics:
        pipeline.plot_metrics()

    if save_metrics:
        pipeline.save_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Train Segmentation Network")
    parser.add_argument("--total_samples", type=int, default=26_000, metavar="", help="Total number of samples to split as train and test. (Values beyond 50_000 will be clamped to 50_000)")
    parser.add_argument("--train_size", type=float, default=0.9, metavar="", help="% Train data size")
    parser.add_argument("--cache_size", type=int, default=55_000, metavar="", help="Number of sample pairs in cache memory")
    parser.add_argument("--img_size", type=int, default=224, metavar="", help="Size of image data for (H and W)")
    parser.add_argument("--volume_first", type=int, default=int(False), choices=[0, 1], metavar="", help="Load entire volume and then slice")
    parser.add_argument("--batch_size", type=int, default=32, metavar="", help="Model training batch size")
    parser.add_argument("--dataset", type=str, default="kits23", choices=["kits23", "lits17"], metavar="", help="Dataset label to specify")
    parser.add_argument("--shuffle", type=int, default=int(True), choices=[0, 1], metavar="", help="Shuffle training data in dataloader for each epoch")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), metavar="", help="Number of dataloader workers")
    parser.add_argument("--loss_func_id", type=str, default="bce", metavar="", help="ID of loss function, one of ('bce', 'focal', 'hybrid')")
    parser.add_argument("--w_bce", type=float, default=0.5, metavar="", help="Weight value of BCE loss when using hybrid loss")
    parser.add_argument("--w_focal", type=float, default=0.5, metavar="", help="Weight value of focal loss when using hybrid loss")
    parser.add_argument("--label_smoothing", type=float, default=0.0, metavar="", help="Label smoothing value")
    parser.add_argument("--loss_reduction", type=str, default="mean", metavar="", help="Loss reduction value ('mean' or 'sum')")
    parser.add_argument("--alpha", type=float, default=0.25, metavar="", help="alpha value for focal or hybrid loss")
    parser.add_argument("--gamma", type=float, default=2.0, metavar="", help="gamma value for focal or hybrid loss")
    parser.add_argument("--input_channels", type=int, default=1, metavar="", help="Model input channels")
    parser.add_argument("--last_fmap_channels", type=int, default=512, metavar="", help="Number of channels for last feature map of encoder (2048 for resnet50 and above)")
    parser.add_argument("--num_classes", type=int, default=3, metavar="", help="Number of segmentation classes")
    parser.add_argument("--block", type=str, default="BasicBlock", metavar="", help="Block type for encoder backbone ('BasicBlock' or 'Bottleneck')")
    parser.add_argument("--block_layers", type=str, default="3,4,6,3", metavar="", help="Number of layers per block")
    parser.add_argument("--decay_lr", type=int, default=int(True), metavar="", help="Decays learning rate via cosine annealing + warm restart")
    parser.add_argument("--T_0", type=int, default=20, metavar="", help="Initial number of iterations for learning rate decay before warm restart")
    parser.add_argument("--epochs", type=int, default=140, metavar="", help="Number of training epochs / cycles")
    parser.add_argument("--lr", type=float, default=2e-4, metavar="", help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, metavar="", help="Model weight decay value")
    parser.add_argument("--min_lr", type=float, default=1e-6, metavar="", help="Minimum learning rate before warm restart")
    parser.add_argument("--T_mult", type=int, default=2, metavar="", help="T_0 multiplier for each warm restart")
    parser.add_argument("--xavier_weight_init", type=int, default=int(True), choices=[0, 1], metavar="", help="Xavier weight initialisation")
    parser.add_argument("--combine_metrics", type=int, default=int(False), choices=[0, 1], metavar="", help="Combine Hausdorff distance and Dice Coefficient with the main loss function (Hausdorff distance is non-differentiable and will only serve as a constant)")
    parser.add_argument("--w_hd", type=float, default=0.5, metavar="", help="Weight value of Hausdorff distance when 'combine_metrics' is True")
    parser.add_argument("--w_dc", type=float, default=0.5, metavar="", help="Weight value of Dice coefficients when 'combine_metrics' is True")
    parser.add_argument("--log_train_step", type=int, default=1, metavar="", help="Training info logging steps")
    parser.add_argument("--verbose", type=int, default=int(True), choices=[0, 1], metavar="", help="Verbosity of training")
    parser.add_argument("--plot_metrics", type=int, default=int(True), choices=[0, 1], metavar="", help="Plots metrics if True")
    parser.add_argument("--save_metrics", type=int, default=int(True), choices=[0, 1], metavar="", help="Save metrics if True")

    args = parser.parse_args()

    METAFILE_PATH = "kits23_metadata.csv" if args.dataset == "kits23" else "lits17_metadata.csv"

    meta_df = pd.read_csv(METAFILE_PATH)

    plain_img_df = meta_df[meta_df["kidney"] != True]
    plain_img_df = plain_img_df[plain_img_df["tumor"] != True]
    plain_img_df = plain_img_df[plain_img_df["cyst"] != True]
    segment_img_df = meta_df[~meta_df.isin(plain_img_df)].dropna()

    plain_img_df.index = range(0, len(plain_img_df))
    segment_img_df.index = range(0, len(segment_img_df))
    data_df = pd.concat([segment_img_df, plain_img_df.sample(frac=10_000/plain_img_df.shape[0])], axis=0)

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
        dataset_label=args.dataset,
        img_size=(args.img_size, args.img_size), 
        cache_size=args.cache_size, 
        train_data_size=args.train_size, 
        volume_first=bool(args.volume_first)
    )

    train(
        train_dataset, 
        eval_dataset, 
        dataset_label=args.dataset,
        batch_size=args.batch_size, 
        shuffle=bool(args.shuffle),
        num_workers=args.num_workers,
        loss_func_id=args.loss_func_id,
        bce_focal_weights=(args.w_bce, args.w_focal),
        label_smoothing=args.label_smoothing if args.label_smoothing != 0 else None,
        loss_reduction=args.loss_reduction,
        alpha=args.alpha,
        gamma=args.gamma,
        input_channels=args.input_channels,
        last_fmap_channels=args.last_fmap_channels,
        num_classes=args.num_classes,
        block=getattr(resnet, args.block),
        block_layers=[int(i) for i in args.block_layers.split(",")],
        decay_lr=bool(args.decay_lr),
        T_0=args.T_0,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        T_mult=args.T_mult,
        xavier_weight_init=bool(args.xavier_weight_init),
        combine_metrics=bool(args.combine_metrics),
        w_hd=args.w_hd,
        w_dc=args.w_dc,
        log_train_step=args.log_train_step,
        verbose=bool(args.verbose),
        plot_metrics=bool(args.plot_metrics),
        save_metrics=bool(args.save_metrics),
    )