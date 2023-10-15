import torch, tqdm, copy, math
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from ._base import BasePipeline
from metrics import dice_coeff, hausdorff_distance
from typing import Tuple


class TrainingPipeline(BasePipeline):
    def __init__(self, **kwargs):
        super(TrainingPipeline, self).__init__(**kwargs)

        # collect metrics in this dictionary
        self._train_metrics_dict = dict(
            loss = [],
            dice_score = [],
            hausdorff_distance = [],
        )
        self._eval_metrics_dict = copy.deepcopy(self._train_metrics_dict)

    def plot_metrics(
            self, 
            mode: str,
            figsize: Tuple[float, float]=(20, 6)):
        
        valid_modes = self._valid_modes()
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        
        _, axs = plt.subplots(1, 3, figsize=figsize)
        axs[0].plot(getattr(self, f"_{mode}_metrics_dict")["loss"])
        axs[0].set_title(f"{mode} Binary Cross Entropy Loss")

        axs[1].plot(getattr(self, f"_{mode}_metrics_dict")["dice_score"])
        axs[1].set_title(f"{mode} Dice Score")

        axs[2].plot(getattr(self, f"_{mode}_metrics_dict")["hausdorff_distance"])
        axs[2].set_title(f"{mode} Hausdorff Distance")

        plt.show()
        print("\n\n")


    def _feed(self, dataloader: DataLoader, mode: str, verbose: bool=False) -> Tuple[float, float, float]:
        assert mode in self._valid_modes(), "Invalid Mode"
        getattr(self.model, mode)()
        pbar_total = math.ceil(len(dataloader.dataset) / dataloader.batch_size)
        loss, dice_score, hausdorff = 0, 0, 0
        
        for idx, (images, gt_masks) in tqdm.tqdm(enumerate(dataloader), total=pbar_total):
            images = images.to(self.device)                          # shape: (N, n_channels, H, W)
            gt_masks = gt_masks.to(self.device, dtype=torch.float32) # shape: (N, n_channels, H, W)    
                        
            pred_masks = self.model(images)
            batch_loss = self.lossfunc(pred_masks, gt_masks)
            batch_dice_score = dice_coeff(pred_masks, gt_masks)
            batch_hausdorff = hausdorff_distance(pred_masks, gt_masks)
            
            if mode == "train":
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            loss += batch_loss.item()
            dice_score += batch_dice_score
            hausdorff += batch_hausdorff

        loss /= (idx + 1)
        dice_score /= (idx + 1)
        hausdorff /= (idx + 1)

        verbosity_label = mode.title()
        if verbose:
            print((
                f"{verbosity_label} Loss: {round(loss, 4)}\
                  \t{verbosity_label} Dice Score: {round(dice_score, 4)}\
                  \t{verbosity_label} Hausdorff Distance: {round(hausdorff, 4)}"
            ))
        getattr(self, f"_{mode}_metrics_dict")["loss"].append(loss)
        getattr(self, f"_{mode}_metrics_dict")["dice_score"].append(dice_score)
        getattr(self, f"_{mode}_metrics_dict")["hausdorff_distance"].append(hausdorff)
        
        return loss, dice_score, hausdorff