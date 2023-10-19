import torch, torchvision
import torch.nn as nn
from typing import Optional, Callable, Iterable, Tuple
from torchvision.models.resnet import ResNet, BasicBlock


#______________________________________________________________________________________________
# BackBone / Encoder Networks                                                                  |
#______________________________________________________________________________________________|
class EncoderHead(ResNet):
    def __init__(
        self, 
        input_channels: int, 
        block: Callable=BasicBlock, 
        block_layers: Iterable[int]=[2, 2, 2, 2]):

        super(EncoderHead, self).__init__(block, block_layers)
        self.block = block
        self.block_layers = block_layers
        self.input_channels = input_channels
            
        self.conv1 = nn.Conv2d(
            self.input_channels, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )

        #delete unwanted layers
        del self.maxpool, self.fc, self.avgpool
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fmap1 = self.conv1(x)
        x = self.bn1(fmap1)
        x = self.relu(x)
        fmap2 = self.layer1(x)
        fmap3 = self.layer2(fmap2)
        fmap4 = self.layer3(fmap3)
        fmap5 = self.layer4(fmap4)
        
        return fmap1, fmap2, fmap3, fmap4, fmap5
    

#______________________________________________________________________________________________
# Segmentation Block                                                                           |
#______________________________________________________________________________________________|
class SegmentationBlock(nn.Module):
    def __init__(
            self, 
            input_channels: int, 
            output_channels: int, 
            upsample_scale: int=2,
            activation: Callable=nn.ReLU):
        
        super(SegmentationBlock, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.upsample_scale = upsample_scale

        self.conv_transpose = nn.ConvTranspose2d(
            self.input_channels, 
            self.input_channels, 
            kernel_size=self.upsample_scale, 
            stride=self.upsample_scale
        )
        self.instance_norm1 = nn.InstanceNorm2d(
            self.conv_transpose.in_channels
        )

        self.conv = nn.Conv2d(
            self.conv_transpose.out_channels, 
            self.output_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.instance_norm2 = nn.InstanceNorm2d(
            self.conv.out_channels
        )
        self.activation = activation()
    
    def forward(self, fmap: torch.Tensor, prev_fmap: Optional[torch.Tensor]=None) -> torch.Tensor:
        output = self.conv_transpose(fmap)
        output = self.instance_norm1(output)
        output = self.conv(output)
        output = self.instance_norm2(output)

        if torch.is_tensor(prev_fmap):
            if prev_fmap.shape != output.shape:
                raise ValueError(
                    f"in {self.__class__.__name__}'s forward pass "
                    "feature maps must be of same shape"
                )
            output += prev_fmap
            
        output = self.activation(output)
        return output
    

#______________________________________________________________________________________________
# Segmentation Head                                                                            |
#______________________________________________________________________________________________|
class SegmentationHead(nn.Module):
    def __init__(
        self, 
        last_fmap_ch: int, 
        output_channels: int):

        super(SegmentationHead, self).__init__()
        
        self.last_fmap_ch = last_fmap_ch
        self.output_channels = output_channels
        self.n_fmap = 5

        self.residual_layers = self._make_layers()

    def forward(self, fmaps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        #fmaps(reversed): fmap4, fmap3, fmap2, fmap1
        fmaps = [fmap for fmap in reversed(fmaps)]

        for idx, m in enumerate(self.residual_layers):
            if idx == 0:                                # at first layer
                output = m(fmaps[idx], fmaps[idx+1])
                continue
            if idx == len(self.residual_layers)-1:      # at final layer
                output = m(output, None)
                continue
            output = m(output, fmaps[idx+1])            # at middle layers
        return output

    def _make_layers(self) -> Iterable[SegmentationBlock]:
        layers: Iterable[SegmentationBlock] = []
        for i in range(self.n_fmap):
            if i == 0: 
                in_ch = self.last_fmap_ch
            else: 
                in_ch = layers[i-1].output_channels

            out_ch = in_ch // 2
            upsample_scale = 2

            if i == self.n_fmap - 3:
                out_ch = in_ch // 2 if self.last_fmap_ch == 2048 else 64
                upsample_scale = 2

            if i == self.n_fmap - 2:
                out_ch = 64
                upsample_scale = 1

            if i == self.n_fmap - 1:
                out_ch = self.output_channels
                upsample_scale = 2

            layer = SegmentationBlock(
                input_channels=in_ch, 
                output_channels=out_ch,
                upsample_scale=upsample_scale, 
            )
            layers.append(layer)

        # change final layer's second instance norm to identity function
        layers[-1].instance_norm2 = nn.Identity()
        layers = nn.ModuleList(layers)
        return layers
    

#______________________________________________________________________________________________
# Segmentation Head                                                                            |
#______________________________________________________________________________________________|
class SegmentationNet(nn.Module):
    def __init__(
            self, 
            input_channels: int,
            *,
            num_classes: int=3,
            block: Callable=BasicBlock, 
            block_layers: Iterable[int]=[3, 4, 6, 3], 
            last_fmap_ch: int=512):
        
        super(SegmentationNet, self).__init__()

        self.encoder_head = EncoderHead(
            input_channels=input_channels, 
            block=block, 
            block_layers=block_layers, 
        )

        self.segmentation_head = SegmentationHead(
            last_fmap_ch=last_fmap_ch, 
            output_channels=num_classes
        )

    def forward(self, x: torch.Tensor):
        fmaps = self.encoder_head(x)
        output = self.segmentation_head(fmaps)
        output = nn.functional.sigmoid(output)
        return output
