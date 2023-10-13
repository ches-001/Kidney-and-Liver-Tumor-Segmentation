import torch, warnings
from monai.metrics import compute_hausdorff_distance

warnings.filterwarnings("ignore")


def dice_coeff(prediction: torch.Tensor, ground_truth: torch.Tensor, epsilon: float=1e-5)->float:
    #input shape: (N, C, H, W)
    prediction = prediction.round()
    ground_truth = ground_truth
    intersection = torch.sum(torch.abs(ground_truth * prediction), dim=(2, 3))
    denominator = ground_truth.sum(dim=(2, 3)) + prediction.sum(dim=(2, 3))
    dice_coeff = ((2 * intersection + epsilon) / (denominator + epsilon)).mean(dim=(0, 1))
    return dice_coeff.cpu().item()


def hausdorff_distance(prediction: torch.Tensor, ground_truth: torch.Tensor):
    #pred shape: (N, C, H, W)
    #target shape: (N, C, H, W)
    prediction = prediction.detach().cpu().round()
    ground_truth = ground_truth.cpu()

    max_dist = max(ground_truth.shape[2:])

    #compute the hausdorff distance for each batch
    batch_distances = compute_hausdorff_distance(
        prediction, 
        ground_truth, 
        distance_metric='euclidean', 
        include_background=True
    )
    batch_distances[batch_distances == torch.inf] = max_dist
    batch_distances[batch_distances != batch_distances] = 0

    #get max distance for each batch and compute mean
    batch_distances = batch_distances.max(dim=1)
    dist = batch_distances[0].mean()
    dist = (dist / max_dist)
    return dist.item()