import torch.nn
from torch import nn

class UNET(nn.Module):

    def __init__(self):
        super(UNET, self).__init__()
        self.Ld1_1 = self.down_conv_block(n=1, nIn=3, nOut=64)
        self.Ld1_2 = self.down_conv_block(n=2, nIn=64, nOut=64)
        self.pool1 = nn.MaxPool2d(2)

        self.Ld2_1 = self.down_conv_block(n=4, nIn=64, nOut=128)
        self.Ld2_2 = self.down_conv_block(n=5, nIn=128, nOut=128)
        self.pool2 = nn.MaxPool2d(2)

        self.Ld3_1 = self.down_conv_block(n=7, nIn=128, nOut=256)
        self.Ld3_2 = self.down_conv_block(n=8, nIn=256, nOut=256)
        self.pool3 = nn.MaxPool2d(2)

        self.Ld4_1 = self.down_conv_block(n=10, nIn=256, nOut=512)
        self.Ld4_2 = self.down_conv_block(n=11, nIn=512, nOut=512)
        self.pool4 = nn.MaxPool2d(2)

        self.L5_1 = self.up_conv_block(n=13, nIn=512, nOut=1024, pad=1)
        self.L5_2 = self.up_conv_block(n=14, nIn=1024, nOut=512, pad=1)
        self.upsample5 = nn.Upsample(scale_factor=2)

        self.Lu4_1 = self.up_conv_block(n=16,nIn=1024,nOut=512)
        self.Lu4_2 = self.up_conv_block(n=17,nIn=512,nOut=256)
        self.upsample4 = nn.Upsample(scale_factor=2)

        self.Lu3_1 = self.up_conv_block(n=19,nIn=512,nOut=256)
        self.Lu3_2 = self.up_conv_block(n=20,nIn=256,nOut=128)
        self.upsample3 = nn.Upsample(scale_factor=2)

        self.Lu2_1 = self.up_conv_block(n=22,nIn=256,nOut=128)
        self.Lu2_2 = self.up_conv_block(n=23,nIn=128,nOut=64)
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.Lu1_1 = self.up_conv_block(n=25,nIn=128,nOut=64)
        self.Lu1_2 = self.up_conv_block(n=26,nIn=64,nOut=1)


    def down_conv_block(self,n,nIn,nOut,pad=0,batch_norm=True ):
        block = nn.Sequential()
        block.add_module(f"down_conv_{n}",nn.Conv2d(nIn,nOut,kernel_size=3, stride=1, padding=pad))
        if(batch_norm):
            block.add_module(f"down_bn_{n}", nn.BatchNorm2d(nOut))
        block.add_module(f"down_act_{n}", nn.ReLU())
        return block

    def up_conv_block(self,n,nIn,nOut,pad=0,batch_norm=True):
        block = nn.Sequential()
        block.add_module(f"up_conv",nn.ConvTranspose2d(nIn,nOut,kernel_size=3,stride=1,padding=pad))
        if(batch_norm):
            block.add_module(f"up_bn_{n}", nn.BatchNorm2d(nOut))
        block.add_module(f"up_act_{n}", nn.ReLU())
        return block

    def forward(self, X):
        X = self.Ld1_1(X)
        X1 = self.Ld1_2(X)
        X = X1
        X = self.pool1(X)

        X = self.Ld2_1(X)
        X2 = self.Ld2_2(X)
        X = X2
        X = self.pool2(X)

        X = self.Ld3_1(X)
        X3 = self.Ld3_2(X)
        X = X3
        X = self.pool3(X)

        X = self.Ld4_1(X)
        X4 = self.Ld4_2(X)
        X = X4
        X = self.pool4(X)

        X = self.L5_1(X)
        X = self.L5_2(X)
        X = self.upsample5(X)
        X = torch.cat([X,X4], dim=1)

        X = self.Lu4_1(X)
        X = self.Lu4_2(X)
        X = self.upsample4(X)
        X = torch.cat([X, X3], dim=1)

        X = self.Lu3_1(X)
        X = self.Lu3_2(X)
        X = self.upsample3(X)
        X = torch.cat([X, X2], dim=1)

        X = self.Lu2_1(X)
        X = self.Lu2_2(X)
        X = self.upsample2(X)
        X = torch.cat([X, X1], dim=1)

        X = self.Lu1_1(X)
        X = self.Lu1_2(X)
        return X



