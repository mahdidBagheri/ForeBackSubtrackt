import torch.nn
from torch import nn

class UNET(nn.Module):

    def __init__(self):
        super(UNET, self).__init__()
        self.down1 = self.down_conv_block(n=1, nIn=3, nOut=8)
        self.down2 = self.down_conv_block(n=2, nIn=8, nOut=32)
        self.up1 = self.up_conv_block(n=4,nIn=32,nOut=8)
        self.up2 = self.up_conv_block(n=5,nIn=8,nOut=1)

    def down_conv_block(self,n,nIn,nOut,batch_norm=True ):
        block = nn.Sequential()
        block.add_module(f"down_conv_{n}",nn.Conv2d(nIn,nOut,kernel_size=5, stride=1, padding=0))
        if(batch_norm):
            block.add_module(f"down_bn_{n}", nn.BatchNorm2d(nOut))
        block.add_module(f"down_act_{n}", nn.ReLU())
        block.add_module(f"downsample_{n}", nn.MaxPool2d(kernel_size=2))
        return block

    def up_conv_block(self,n,nIn,nOut,batch_norm=True):
        block = nn.Sequential()
        block.add_module(f"upsampling_{n}", nn.Upsample(scale_factor=2,mode='bilinear'))
        block.add_module(f"up_conv",nn.ConvTranspose2d(nIn,nOut,kernel_size=5,stride=1,padding=0))
        if(batch_norm):
            block.add_module(f"up_bn_{n}", nn.BatchNorm2d(nOut))
        block.add_module(f"up_act_{n}", nn.ReLU())

        return block

    def forward(self, X):
        X = self.down1(X)
        X = self.down2(X)
        X = self.up1(X)
        X = self.up2(X)
        return X



