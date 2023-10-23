import os, torch, argparse, tqdm
import nibabel as nib
import numpy as np
import pyvista as pv
from torchvision.models import resnet
from typing import Optional, Union, Tuple
from seg_network import SegmentationNet
import matplotlib.pyplot as plt
from transforms import img_normalize, img_resize
from metrics import dice_coeff, hausdorff_distance

os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor


def visualise_slice(_slice: np.ndarray, output: torch.Tensor, target: Optional[torch.Tensor]=None):
    n_plots = 2
    if isinstance(target, np.ndarray): n_plots += 1

    fig, axs = plt.subplots(1, n_plots, figsize=(15, 5))

    axs[0].imshow(_slice, cmap="gray")
    axs[0].set_title("Input CT / MRI slice")

    axs[1].imshow(_slice, cmap="gray")
    axs[1].imshow(output, alpha=0.3)
    axs[1].set_title("Predicted Segment(s)")

    if isinstance(target, np.ndarray): 
        axs[2].imshow(_slice, cmap="gray")
        axs[2].imshow(target, alpha=0.3)
        axs[2].set_title("Target Segment(s)")
    plt.show()
        

def visualise_volume(volume: np.ndarray, output: np.ndarray, target: Optional[np.ndarray]=None):
    vol_grid = pv.ImageData()
    vol_grid.dimensions = volume.shape
    vol_grid.point_data["values"] = volume.ravel(order="F")

    output_grid = pv.ImageData()
    output_grid.dimensions = output.shape
    output_grid.point_data["values"] = output.ravel(order="F")

    n_plots = 2
    if isinstance(target, np.ndarray):
        target_grid = pv.ImageData()
        target_grid.dimensions = target.shape
        target_grid.point_data["values"] = target.ravel(order="F")
        n_plots += 1

    pl = pv.Plotter(shape=(1, n_plots), notebook=False, border=True, window_size=(1536, 512))
    pl.subplot(0, 0)
    pl.add_volume(vol_grid, cmap="gray")
    pl.add_title("Input CT / MRI volume", font_size=7)

    pl.subplot(0, 1)
    pl.add_volume(vol_grid, cmap="gray")
    pl.add_volume(output_grid)
    pl.add_title("Predicted segment(s)", font_size=7)
    
    if isinstance(target, np.ndarray):
        pl.subplot(0, 2)
        pl.add_volume(vol_grid, cmap="gray")
        pl.add_volume(target_grid)
        pl.add_title("Target segment(s)", font_size=7)

    pl.show()


def to_binary_mask(segment: torch.Tensor) -> torch.Tensor:
    binary_segment = torch.zeros((NUM_CLASSES, *IMG_SIZE), dtype=segment.dtype)

    for i, label in enumerate(range(1, NUM_CLASSES+1)):
        template = segment.clone()
        template[template != label] = 0
        template[template == label] = 1
        binary_segment[i] = template
    return binary_segment


def to_label_segment(segment: torch.Tensor) -> torch.Tensor:
    label_segment = torch.zeros(segment.shape[1:], dtype=segment.dtype)
    for channel in range(0, segment.shape[0]):
        label_segment[segment[channel]==1] = channel+1
    return label_segment


def evaluate(
        model: SegmentationNet, 
        img: torch.Tensor, 
        target: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
    
    with torch.no_grad():
        img = img.to(device=os.getenv("DEVICE"))
        output = model(img)
        output = output.round()

    dice_score, hausdorff = None, None
    if torch.is_tensor(target):
        target = target.to(device=os.getenv("DEVICE"))
        dice_score = dice_coeff(output, target.unsqueeze(dim=0)).item()
        hausdorff = hausdorff_distance(output, target.unsqueeze(dim=0)).item()
        dice_score = round(dice_score, 4)
        hausdorff = round(hausdorff, 4)

    output = output.squeeze(dim=0)
    return output.cpu(), dice_score, hausdorff


def evaluate_slice(
    model: SegmentationNet, 
    case_vol: Union[str, np.ndarray], 
    slice_idx: int, 
    segment_vol: Optional[Union[str, np.ndarray]]=None, 
    plot: bool=False) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
    
    load_slice = (
        lambda path : 
        nib.load(path).dataobj[slice_idx]
        if SLICE_DIM == "first" 
        else nib.load(path).dataobj[:, :, slice_idx]
    )

    get_slice = (
        lambda volume : 
        volume[slice_idx] 
        if SLICE_DIM == 'first' 
        else volume[:, :, slice_idx]
    )

    img = load_slice(case_vol) if isinstance(case_vol, str) else get_slice(case_vol)
    og_size = img.shape

    if not img.flags["WRITEABLE"]:
        img = img.copy()
    img = torch.from_numpy(img).unsqueeze(dim=0)
    img = img_resize(img, size=IMG_SIZE)
    img = img_normalize(img)
    img = img.to(dtype=torch.float32).unsqueeze(dim=0)

    segment = None
    if segment_vol is not None:
        segment = load_slice(segment_vol) if isinstance(segment_vol, str) else get_slice(segment_vol)
        if not segment.flags["WRITEABLE"]:
            segment = segment.copy()
        segment = torch.from_numpy(segment).unsqueeze(dim=0)
        segment = img_resize(segment, size=IMG_SIZE)
        segment = segment.to(dtype=torch.float32).unsqueeze(dim=0)
        segment = to_binary_mask(segment)

    output, dice_score, hausdorff = evaluate(model, img, segment)
    output = img_resize(output, size=og_size)
    output = to_label_segment(output)

    if plot:
        img = img_resize(img, size=og_size)
        if torch.is_tensor(segment):
            segment = img_resize(segment.squeeze(dim=0), size=og_size)
            segment = to_label_segment(segment)

        visualise_slice(
            to_numpy(img.squeeze()), 
            to_numpy(output), 
            to_numpy(segment),
        )
    return output, dice_score, hausdorff


def evaluate_volume(
        model: SegmentationNet, 
        case_path: str, 
        segment_path: Optional[str]=None, 
        plot: bool=False) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
    
    case_vol = nib.load(case_path).get_fdata()
    
    segment_vol = None
    if segment_path is not None:
        segment_vol = nib.load(segment_path).get_fdata()

    n_slices = case_vol.shape[0] if SLICE_DIM == "first" else case_vol.shape[-1]

    avg_dice_score, avg_hausdorff = 0, 0
    predicted_vol = torch.zeros(case_vol.shape, dtype=torch.float32, device="cpu")

    for slice_idx in tqdm.tqdm(range(0, n_slices), total=n_slices):
        output, dice_score, hausdorff = evaluate_slice(
            model, 
            case_vol=case_vol, 
            slice_idx=slice_idx,
            segment_vol=segment_vol, 
        )

        if SLICE_DIM == "first":
            predicted_vol[slice_idx] = output
        else:
            predicted_vol[:, :, slice_idx] = output

        if dice_score is not None:
            avg_dice_score += dice_score / n_slices
            avg_hausdorff += hausdorff / n_slices

    if plot:
        visualise_volume(case_vol, to_numpy(predicted_vol), segment_vol)
    avg_dice_score = round(avg_dice_score, 4)
    avg_hausdorff = round(avg_hausdorff, 4)
    return predicted_vol, avg_dice_score, avg_hausdorff



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tumor Segmentation Network")
    parser.add_argument("case_path", type=str, metavar="", help="Path to case volume")
    parser.add_argument("model_type", type=str, choices=["kidney", "liver"], metavar="", help="Specify the model type based on the dataset it was trained on ('kidney' or 'liver')")
    parser.add_argument("slice_dim", type=str, choices=["first", "last"], metavar="", help="Specify what axes of the volume corresponds to slice axes ('first' or 'last')")

    parser.add_argument("--slice_idx", default="", type=str, metavar="", help="if provided, inference is run on slice rather than the entire volume")
    parser.add_argument("--segment_path", default="", type=str, metavar="", help="Path to corresponding segmentation volume of case")
    parser.add_argument("--volume_first", type=int, default=int(False), choices=[0, 1], metavar="", help="Load entire volume and then slice")
    parser.add_argument("--img_size", type=int, default=224, metavar="", help="Size of image data for (H and W)")
    parser.add_argument("--input_channels", type=int, default=1, metavar="", help="Model input channels")
    parser.add_argument("--last_fmap_channels", type=int, default=512, choices=[512, 2048], metavar="", help="Number of channels for last feature map of encoder (2048 for resnet50 and above)")
    parser.add_argument("--block", type=str, default="BasicBlock", choices=["BasicBlock", "Bottleneck"], metavar="", help="Block type for encoder backbone ('BasicBlock' or 'Bottleneck')")
    parser.add_argument("--block_layers", type=str, default="3,4,6,3", metavar="", help="Number of layers per block")
    args = parser.parse_args()    

    MODEL_PATH, NUM_CLASSES = (
        ("saved_model/kits23/model.pth.tar", 3) 
        if args.model_type == "kidney" else 
        ("saved_model/lits17/model.pth.tar", 2)
    ) 
    SLICE_DIM = args.slice_dim
    IMG_SIZE = (args.img_size, args.img_size)

    model = SegmentationNet(
        args.input_channels, 
        num_classes=NUM_CLASSES,
        block=getattr(resnet, args.block),
        block_layers=[int(i) for i in args.block_layers.split(",")],
        last_fmap_channels=args.last_fmap_channels,
    )
    model_weights = torch.load(MODEL_PATH, map_location=os.getenv("DEVICE"))
    model.load_state_dict(model_weights["network_params"])
    model.to(device=os.getenv("DEVICE"))
    model.eval()

    # Run:
    #"python evaluate kits23/dataset/case_00456/imaging.nii.gz" kidney first
    #"python evaluate lits17/dataset/volume-24.nii" liver last --segment_path="lits17/dataset/segmentation-24.nii"

    if args.slice_idx:
        output, dice_score, hausdorff = evaluate_slice(
            model,
            case_vol=args.case_path, 
            slice_idx=int(args.slice_idx),
            segment_vol=args.segment_path if args.segment_path else None,
            plot=True,
        )

    else:
        output, dice_score, hausdorff = evaluate_volume(
            model, 
            case_path=args.case_path, 
            segment_path=args.segment_path if args.segment_path else None,
            plot=True,
        )

    if args.segment_path:
        print(f"Dice Score: {dice_score}")
        print(f"Hausdorff Distance: {hausdorff}")