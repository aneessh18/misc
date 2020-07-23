import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels,kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv

def crop_image(current_tensor : torch.Tensor, target_tensor) -> torch.Tensor: #center crop
    current_size = current_tensor.size()[2]
    target_size = target_tensor.size()[2]
    diff = current_size-target_size
    diff = diff//2
    return current_tensor[:,:,diff:current_size-diff,diff:current_size-diff]

def conv_transpose(in_channels, out_channels):
    return nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2
    )
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool_2d = nn.MaxPool2d(kernel_size=2,stride=2)
        self.de_conv1 = double_conv(1,64)
        self.de_conv2 = double_conv(64,128)
        self.de_conv3 = double_conv(128,256)
        self.de_conv4 = double_conv(256,512)
        self.de_conv5 = double_conv(512,1024)
        self.up_trans1 = conv_transpose(1024,512)
        self.up_conv1 = double_conv(1024,512)
        self.up_trans2 = conv_transpose(512,256)
        self.up_conv2 = double_conv(512,256)
        self.up_trans3 = conv_transpose(256,128)
        self.up_conv3 = double_conv(256,128)
        self.up_trans4 = conv_transpose(128,64)
        self.up_conv4 = double_conv(128,64)
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=2,kernel_size=1)
    def forward(self, image:torch.Tensor):
        x1 = self.de_conv1(image) #
        x2 = self.max_pool_2d(x1)
        x3 = self.de_conv2(x2) #
        x4 = self.max_pool_2d(x3)
        x5 = self.de_conv3(x4) #
        x6 = self.max_pool_2d(x5)
        x7 = self.de_conv4(x6) #
        x8 = self.max_pool_2d(x7)
        x9 = self.de_conv5(x8)

        x = self.up_trans1(x9)
        y = crop_image(x7, x)
        x = self.up_conv1(torch.cat([y,x],dim=1))

        x = self.up_trans2(x)
        y = crop_image(x5,x)
        x = self.up_conv2(torch.cat([y,x],1))
            
        x = self.up_trans3(x)
        y = crop_image(x3,x)
        x = self.up_conv3(torch.cat([y,x],1))
        
        x = self.up_trans4(x)
        y = crop_image(x1,x)
        x = self.up_conv4(torch.cat([y,x],1))
        
        out = self.conv5(x)
        print(out.size())
        return out
        

        


if __name__ == "__main__":
    model = UNet()
    image = torch.rand(1,1,572,572)
    model(image)    