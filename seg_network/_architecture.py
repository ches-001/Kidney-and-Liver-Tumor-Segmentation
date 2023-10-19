import torch, torchvision
import torch.nn as nn
from typing import Optional, Callable, Iterable, Tuple
from torchvision.models.resnet import ResNet, BasicBlock


#______________________________________________________________________________________________
# BackBone / Encoder Networks                                                                  |
#______________________________________________________________________________________________|
class EncoderHead(ResNet):
    def __init__(
        self, input_channels: int, block: Callable=BasicBlock, block_layers: Iterable[int]=[3, 4, 6, 3]):
      
        self.block = block
        self.block_layers = block_layers
        
        super(EncoderHead, self).__init__(self.block, self.block_layers)
        
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(
            self.input_channels, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False)
        
        self.conv2 = nn.Conv2d(
            64, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3),
            bias=False)

        #delete unwanted layers
        del self.maxpool, self.fc, self.avgpool
        
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fmap1 = self.conv1(x)
        x = self.conv2(fmap1)
        x = self.bn1(x)
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
    def __init__(self, input_channels: int, output_channels: int, upsample_scale: int=2, activation: Callable=nn.ReLU):
        super(SegmentationBlock, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.upsample_scale = upsample_scale

        self.transpose = nn.ConvTranspose2d(
            self.input_channels, self.input_channels, 
            kernel_size=self.upsample_scale, stride=self.upsample_scale)
        
        self.instance_norm1 = nn.InstanceNorm2d(self.transpose.in_channels)

        self.conv = nn.Conv2d(
            self.transpose.out_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        
        self.instance_norm2 = nn.InstanceNorm2d(self.conv.out_channels)
        
        self.activation = activation()

    
    def forward(self, fmap: torch.Tensor, prev_fmap: Optional[torch.Tensor]=None) -> torch.Tensor:
        output = self.transpose(fmap)
        output = self.instance_norm1(output)
        output = self.conv(output)
        output = self.instance_norm2(output)

        if torch.is_tensor(prev_fmap):
            assert (prev_fmap.shape == output.shape), 'feature maps must be of same shape'
            output += prev_fmap
            
        output = self.activation(output)
        return output

    

#______________________________________________________________________________________________
# Segmentation Head                                                                            |
#______________________________________________________________________________________________|
class SegmentationHead(nn.Module):
    def __init__(
        self, last_fmap_channels: int, num_classes: int):
        super(SegmentationHead, self).__init__()
        
        self.last_fmap_channels = last_fmap_channels
        self.first_fmap_channels = 64
        self.num_classes = num_classes

        self.residual_layers = self.make_layers()

        self.final_layer = SegmentationBlock(
            input_channels=self.residual_layers[-1].output_channels, 
            output_channels=self.num_classes,
            activation=nn.Sigmoid,
        )
        self.final_layer.instance_norm2 = nn.Identity()


    def forward(self, *fmaps: torch.Tensor) -> torch.Tensor:
        #fmaps(reversed): fmap5, fmap4, fmap3, fmap2, fmap1
        fmaps = [fmap for fmap in reversed(fmaps)]
        for idx, m in enumerate(self.residual_layers):
            if idx == 0:
                output = m(fmaps[idx], fmaps[idx+1])
                continue
            output = m(output, fmaps[idx+1])
        
        output = self.final_layer(output)
        return output

      
    def make_layers(self, n: int=4):
        layers = []
        for i in range(n):
            if i == 0: in_ch = self.last_fmap_channels
            else: in_ch = layers[i-1].output_channels

            out_ch = in_ch // 2 if i != (n-1) else self.first_fmap_channels
            layer = SegmentationBlock(in_ch, out_ch)
            layers.append(layer)

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
            last_fmap_channels: int=512):
        
        super(SegmentationNet, self).__init__()

        self.encoder_head = EncoderHead(
            input_channels=input_channels, 
            block=block, 
            block_layers=block_layers, 
        )

        self.segmentation_head = SegmentationHead(
            last_fmap_channels=last_fmap_channels, 
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmaps = self.encoder_head(x)
        output = self.segmentation_head(*fmaps)
        return output
