import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, light_conv, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # group1 = mid_channels if light_conv else 1
        conv2 = 1 if light_conv else 3
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=conv2, padding=conv2//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class PixelShuffleDown(nn.Module):
    """Downscaling with pixel shuffle then double conv"""

    def __init__(self, in_channels, out_channels, light_conv):
        super().__init__()
        # self.conv1 = DoubleConv(in_channels * 4, out_channels, light_conv)
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x : (B, C, H, W)
        x = F.pixel_unshuffle(x, 2) # (B, C*4, H/2, W/2)
        return self.conv1(x)

class MaxPoolDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, light_conv):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, light_conv)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class AvgPoolDown(nn.Module):
    """Downscaling with avgpool then double conv"""

    def __init__(self, in_channels, out_channels, light_conv):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels, light_conv)
        )

    def forward(self, x):
        return self.avgpool_conv(x)

class TransConvUp(nn.Module):
    """Upscaling by transpose convolution then double conv"""

    def __init__(self, in_channels, skip_channels, out_channels, light_conv, skip):
        super().__init__()
        self.skip = skip
        assert skip in ["add", "concat"]
        if skip != "add":
            mid_channels = in_channels // 2
        else:
            mid_channels = skip_channels
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        
        extra_channels = skip_channels if skip == "concat" else 0
        self.conv = DoubleConv(mid_channels + extra_channels, out_channels, light_conv=light_conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if self.skip == "concat":
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1 + x2
        return self.conv(x)

class BilinearUp(nn.Module):
    """Upscaling by bilinear upsampling then double conv"""

    def __init__(self, in_channels, skip_channels, out_channels, light_conv, skip):
        super().__init__()
        self.skip = skip

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        )
        assert skip in ["add", "concat"]
        extra_channels = skip_channels if skip == "concat" else 0
        self.conv = DoubleConv(skip_channels + extra_channels, out_channels, light_conv=light_conv)

    def forward(self, x1, x2):
        print(x1.shape)
        x1 = self.up(x1)
        
        if self.skip == "concat":
            x = torch.cat([x2, x1], dim=1)
        else:
            print(x1.shape, x2.shape)
            x = x1 + x2
        return self.conv(x)

class PixelShuffleUp(nn.Module):
    """Upscaling with pixel shuffle"""
    """no skip connection"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv1 = DoubleConv(in_channels, in_channels*4, light_conv=False, mid_channels=in_channels*2)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels*4, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(in_channels*4),
        #     nn.ReLU(inplace=True)
        # )
        self.out_channels = out_channels
        if out_channels == 1:
            self.out = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, out_channels, kernel_size=1)
            )

        else:
            self.out = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

    def forward(self, x):
        # B, C, H, W = x.shape 
    
        x = F.interpolate(x, scale_factor=2, mode='bilinear') # (B, C, H*2, W*2)
        return self.out(x)
        # x = self.conv1(x) # (B, C*4, H, W)
        # k = torch.arange(C) * 4
        # base = x[:, k, ...] # first channel for every 2*2 pixel, used to bilinear up
        # base = F.interpolate(base, scale_factor=2, mode='bilinear') # (B, C, H*2, W*2)
        # x = x.clone()
        # x[:, k, ...] = 0
        # x = F.pixel_shuffle(x, 2) # (B, C, H*2, W*2)
        # return base + x

class UNet(nn.Module):
    def __init__(self, in_channels=3, filter_size=1, layer_num=4, dilation=0, down="maxpool", up="transconv",
                 skip="concat", light_conv=False, less_channel=False):
        """
        Args:
            down (str, optional): "maxpool" or "avgpool"
            up (str, optional): "transconv" or "binlinear
            skip (str, optional): "concat" or "add"
            light_conv (bool, optional): Use 3*3 conv + 1*1 conv instead of two 3*3 conv
            less_channel (bool, optional): shrink number of channels
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.filter_size = filter_size
        self.dilation = dilation
        self.layer_num = layer_num
        self.skip = skip
        self.light_conv = light_conv
        self.less_channel = less_channel

        if dilation > 0:
            assert filter_size in [3, 5]
        
        assert self.layer_num in [3, 4, 5]
        if self.layer_num == 3:
            channels = [32, 64, 128, 256] if not less_channel else [16, 32, 64, 128]
        elif self.layer_num == 4:
            channels = [32, 32, 64, 128, 256] if not less_channel else [16, 16, 32, 64, 128]
        elif self.layer_num == 5:
            channels = [32, 32, 64, 128, 256, 512] if not less_channel else [16, 16, 32, 64, 128, 256]

        print(f"layer num = {self.layer_num}, effective receptive field is {3 * (1 << self.layer_num)}")

        if down == "maxpool":
            Down = MaxPoolDown
        elif down == "avgpool":
            Down = AvgPoolDown
        else:
            assert False
        
        if up == "transconv":
            Up = TransConvUp
        elif up == "bilinear":
            Up = BilinearUp
        else:
            assert False

        self.inc = DoubleConv(in_channels, channels[0], light_conv)

        self.ups = nn.ParameterList()
        self.downs = nn.ParameterList()
        for i in range(1, len(channels)):
            self.downs.append(Down(channels[i - 1], channels[i], light_conv))
            self.ups.append(Up(in_channels=channels[i], skip_channels=channels[i - 1], out_channels=channels[i - 1], light_conv=light_conv, skip=skip))

        self.outc = nn.Conv2d(channels[0], (dilation + 1) * filter_size * filter_size, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        res = []
        for down in self.downs:
            res.append(x)
            x = down(x)
        for skip, up in zip(reversed(res), reversed(self.ups)):
            x = up(x, skip)

        return self.outc(x)

class PixelShuffleUNet(nn.Module):
    def __init__(self, in_channels=3, filter_size=1, layer_num=4, dilation=0, down="maxpool", up="transconv",
                 skip="concat", light_conv=False, less_channel=False):
        """
        Args:
            down (str, optional): "maxpool" or "avgpool"
            up (str, optional): "transconv" or "binlinear
            skip (str, optional): "concat" or "add"
            light_conv (bool, optional): Use 3*3 conv + 1*1 conv instead of two 3*3 conv
            less_channel (bool, optional): shrink number of channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.filter_size = filter_size
        self.dilation = dilation
        self.layer_num = layer_num
        self.skip = skip
        self.light_conv = light_conv
        self.less_channel = less_channel

        if dilation > 0:
            assert filter_size in [3, 5]
        
        assert self.layer_num in [3, 4, 5]
        if self.layer_num == 3:
            channels = [64, 128, 256] if not less_channel else [16, 64, 128]
        elif self.layer_num == 4:
            channels = [64, 64, 128, 256] if not less_channel else [32, 32, 64, 128]
        elif self.layer_num == 5:
            channels = [64, 64, 128, 256, 512] if not less_channel else [32, 32, 64, 128, 256]
        # if self.layer_num == 3:
        #     channels = [32, 64, 128, 256] if not less_channel else [16, 32, 64, 128]
        # elif self.layer_num == 4:
        #     channels = [32, 32, 64, 128, 256] if not less_channel else [16, 16, 32, 64, 128]
        # elif self.layer_num == 5:
        #     channels = [32, 32, 64, 128, 256, 512] if not less_channel else [16, 16, 32, 64, 128, 256]

        print(f"layer num = {self.layer_num}, effective receptive field is {3 * (1 << self.layer_num)}")

        if down == "maxpool":
            Down = MaxPoolDown
        elif down == "avgpool":
            Down = AvgPoolDown
        else:
            assert False
        
        if up == "transconv":
            Up = TransConvUp
        elif up == "bilinear":
            Up = BilinearUp
        else:
            assert False

        self.inc = PixelShuffleDown(in_channels, channels[0], light_conv)

        self.ups = nn.ParameterList()
        self.downs = nn.ParameterList()
        for i in range(1, len(channels)):
            self.downs.append(Down(channels[i - 1], channels[i], light_conv))
            self.ups.append(Up(in_channels=channels[i], skip_channels=channels[i - 1], out_channels=channels[i - 1], light_conv=light_conv, skip=skip))
        
        self.outc = PixelShuffleUp(channels[0], (dilation + 1) * filter_size * filter_size)

    def forward(self, x):
        x = self.inc(x)
        res = []
        for down in self.downs:
            res.append(x)
            x = down(x)
        for skip, up in zip(reversed(res), reversed(self.ups)):
            x = up(x, skip)

        return self.outc(x)

if __name__ == "__main__":
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    model = PixelShuffleUNet(in_channels=8, filter_size=1, dilation=0, layer_num=4, down='maxpool', up="transconv", 
                             skip="add", light_conv=False, less_channel=False).to(device)
    print("number of parameters:", sum(p.numel() for p in model.parameters()))
    # data = torch.rand((2, 24, 1024, 2048)).to(device)
    # output = model(data)
    # print(output.shape)