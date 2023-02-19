import torch.nn
from torch import nn

class UNET(nn.Module):

    def __init__(self):
        super(UNET, self).__init__()
        self.down1 = self.down_conv_block(n=1, nIn=3, nOut=8, is_pooling=True)
        self.down2 = self.down_conv_block(n=2, nIn=8, nOut=32, is_pooling=False)
        self.down3 = self.down_conv_block(n=3, nIn=32, nOut=128, is_pooling=True)
        self.down4 = self.down_conv_block(n=4, nIn=128, nOut=512, is_pooling=False)
        self.up1 = self.up_conv_block(n=4,nIn=512,nOut=128, is_upsample=False)
        self.up2 = self.up_conv_block(n=5,nIn=128,nOut=32, is_upsample=True)
        self.up3 = self.up_conv_block(n=6,nIn=32,nOut=8, is_upsample=False)
        self.up4 = self.up_conv_block(n=7,nIn=8,nOut=1, is_upsample=True, batch_norm=False)
        self.softmax = nn.Softmax(dim=1)

    def down_conv_block(self,n,nIn,nOut,is_pooling,batch_norm=True ):
        block = nn.Sequential()
        block.add_module(f"down_conv_{n}",nn.Conv2d(nIn,nOut,kernel_size=5, stride=1, padding=0))
        if(batch_norm):
            block.add_module(f"down_bn_{n}", nn.BatchNorm2d(nOut))
        block.add_module(f"down_act_{n}", nn.ReLU())
        if(is_pooling):
            block.add_module(f"downsample_{n}", nn.MaxPool2d(kernel_size=2))
        return block

    def up_conv_block(self,n,nIn,nOut,is_upsample,batch_norm=True):
        block = nn.Sequential()
        if(is_upsample):
            block.add_module(f"upsampling_{n}", nn.Upsample(scale_factor=2,mode='bilinear'))
        block.add_module(f"up_conv",nn.ConvTranspose2d(nIn,nOut,kernel_size=5,stride=1,padding=0))
        if(batch_norm):
            block.add_module(f"up_bn_{n}", nn.BatchNorm2d(nOut))
        block.add_module(f"up_act_{n}", nn.ReLU())

        return block

    def forward(self, X):
        X = self.down1(X)
        X = self.down2(X)
        X = self.down3(X)
        X = self.down4(X)
        X = self.up1(X)
        X = self.up2(X)
        X = self.up3(X)
        X = self.up4(X)
        return X



