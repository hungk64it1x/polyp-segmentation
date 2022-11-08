import torch
import torch.nn as nn

class FBC(nn.Module):

    def __init__(self, channel=512, channel_out=128):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv2d(channel,channel_out,kernel_size=(1,1))

    def forward(self, x):
        b, c, h, w = x.size()
        residual = x
        channel_wv = self.ch_wv(x) #bs,c//2,h,w
        channel_wq = self.ch_wq(x) #bs,1,h,w
        channel_wv = channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq = channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out = channel_weight*x
        
        return self.conv(residual + channel_out)

if __name__ == '__main__':
    x = torch.rand(2, 256, 64, 64)
    att = ParallelPolarizedSelfAttention(256, 128)
    print(att(x).shape)