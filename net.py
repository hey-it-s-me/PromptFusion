import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numpy as np #-----
from PIL import Image #-----
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_grayscale
from pycontourlet.pycontourlet4d.pycontourlet import batch_multi_channel_pdfbdec
from torch.utils.data import Dataset
import kornia
def stack_same_dim(x):
    """Stack a list/dict of 4D tensors of same img dimension together."""
    # Collect tensor with same dimension into a dict of list
    # 这段代码实现了一种将具有相同图像维度的张量列表或字典堆叠在一起的功能，使得用户可以更方便地处理具有相同图像尺寸的数据。
    output = {}
    
    # Input is list
    if isinstance(x, list):
        for i in range(len(x)):
            if isinstance(x[i], list):
                for j in range(len(x[i])):
                    shape = tuple(x[i][j].shape)
                    if shape in output.keys():
                        output[shape].append(x[i][j])
                    else:
                        output[shape] = [x[i][j]]
            else:
                shape = tuple(x[i].shape)
                if shape in output.keys():
                    output[shape].append(x[i])
                else:
                    output[shape] = [x[i]]
    else:
        for k in x.keys():
            shape = tuple(x[k].shape[2:4])
            if shape in output.keys():
                output[shape].append(x[k])
            else:
                output[shape] = [x[k]]
    
    # Concat the list of tensors into single tensor
    for k in output.keys():
        output[k] = torch.cat(output[k], dim=1)
        
    return output

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class ContourletCNN(nn.Module):
    # 接受一些参数，包括 num_classes（分类的类别数）、input_dim（输入图像的维度，默认为 3 通道，224x224）、
    # n_levs（轮廓波分解的级别，默认为 [0, 3, 3, 3]）、variant（模型变体，默认为 "SSF"）和 spec_type（轮廓波变换的类型，默认为 "all"）。

    def __init__(self, input_dim=(1, 224, 224), n_levs=[0, 3, 3, 3], variant="SSF", spec_type="all"):
        super(ContourletCNN, self).__init__()
        # Model hyperparameters
        # self.num_classes = num_classes
        
        self.input_dim = input_dim
        self.n_levs = n_levs
        self.variant = variant
        self.spec_type = spec_type
        
        # Conv layers parameters
        # 这些行定义了卷积层的参数，包括每个卷积层的输出通道数。
        out_conv_1 = 64
        out_conv_2 = 64
        out_conv_3 = 128
        in_conv_4 = 128
        out_conv_4 = 128
        out_conv_5 = 256
        in_conv_6 = 256
        out_conv_6 = 256
        out_conv_7 = 512
        if spec_type == "avg":
            if variant == "origin":
                in_conv_2 = out_conv_1 + (2**n_levs[3]) // 2
                in_conv_3 = out_conv_2 + (2**n_levs[2]) // 2
                in_conv_5 = out_conv_4 + (2**n_levs[1]) // 2
                in_conv_7 = out_conv_6 + 4
            else:
                in_conv_2 = out_conv_1 + 2**n_levs[3]
                in_conv_3 = out_conv_2 + 2**n_levs[2]
                in_conv_5 = out_conv_4 + 2**n_levs[1]
                in_conv_7 = out_conv_6 + 4
        else:
            if variant == "origin":
                in_conv_2 = out_conv_1 + (2**n_levs[3] // 2) * input_dim[0]
                in_conv_3 = out_conv_2 + (2**n_levs[2] // 2) * input_dim[0]
                in_conv_5 = out_conv_4 + (2**n_levs[1] // 2) * input_dim[0]
                in_conv_7 = out_conv_6 + 4 * input_dim[0]
            else:
                in_conv_2 = out_conv_1 + 2**n_levs[3] * input_dim[0]
                in_conv_3 = out_conv_2 + 2**n_levs[2] * input_dim[0]
                in_conv_5 = out_conv_4 + 2**n_levs[1] * input_dim[0]
                in_conv_7 = out_conv_6 + 4 * input_dim[0]
                
        # Conv layers
        #这些行定义了模型中的卷积层，并使 nn.Conv2d创建了卷积层对象。
        self.conv_1 = nn.Conv2d(input_dim[0], out_conv_1, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_conv_2, out_conv_2, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_conv_3, out_conv_3, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(in_conv_4, out_conv_4, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(in_conv_5, out_conv_5, kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv2d(in_conv_6, out_conv_6, kernel_size=3, stride=2, padding=1)
        self.conv_7 = nn.Conv2d(in_conv_7, out_conv_7, kernel_size=3, stride=1, padding=1)
        
        self.Up3 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv3 = conv_block(ch_in=512, ch_out=256)

        self.Up2 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv2 = conv_block(ch_in=256, ch_out=128)

        self.Up1 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv1 = conv_block(ch_in=128, ch_out=64)

        self.Up0 = up_conv(ch_in=64, ch_out = input_dim[0])
        self.Up_conv0 = conv_block(ch_in=64, ch_out= input_dim[0])
        self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
    def __pdfbdec(self, x, method="resize"):
        
        # Convert to from N-D channels to single channel by averaging
        if self.spec_type == 'avg':
            imgs = []
            # Iterate each image in a batch
            for i in range(x.shape[0]):
                # Convert to PIL and image and to grayscale image
                img = transforms.ToPILImage()(x[i])
                img = to_grayscale(img)
                imgs.append(img)
            # Restack and convert back to PyTorch tensor
            x = torch.from_numpy((np.expand_dims(np.stack(imgs, axis=0), axis=1)))

        # Obtain coefficients
        coefs = batch_multi_channel_pdfbdec(x=x, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0,3,3,3], device=self.device)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Resize or splice
        if method == "resize":
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get maximum dimension (height or width)
                    max_dim = int(np.max((k[2], k[3])))
                    # Resize the channels
                    trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                    coefs[k] = trans(coefs[k])
        else:
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get minimum dimension (height or width)
                    min_dim = int(np.argmin((k[2], k[3]))) + 2
                    # Splice alternate channels (always even number of channels exist)
                    coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Change coefs's key to number (n-1 to 0), instead of dimension
        for i, k in enumerate(coefs.copy()):
            idx = len(coefs.keys()) - i - 1
            coefs[idx] = coefs.pop(k)

        # Get statistical features (mean and std) for each image
        sfs = []
        for k in coefs.keys():
            sfs.append(coefs[k].mean(dim=[2, 3]))
            sfs.append(coefs[k].std(dim=[2, 3]))
        sfs = torch.cat(sfs, dim=1)

        return coefs, sfs
        
    def forward(self, x):
        x = kornia.resize(x,(224,224))
        # Perform PDFB decomposition to obtain the coefficients and it's statistical features
        if self.variant == "origin":
            coefs, _ = self.__pdfbdec(x, method="splice")
        else:
            coefs, sfs = self.__pdfbdec(x, method="resize")
        
        # AlexNet backbone convolution layers
        x1 = self.conv_1(x)

        x2 = self.conv_2(torch.cat((x1, coefs[0].to(self.device)), 1))
        x3 = self.conv_3(torch.cat((x2, coefs[1].to(self.device)), 1))
        x4 = self.conv_4(x3)
        x5 = self.conv_5(torch.cat((x4, coefs[2].to(self.device)), 1))
        x6 = self.conv_6(x5)
        x7 = self.conv_7(torch.cat((x6, coefs[3].to(self.device)), 1))
        d3 = self.Up3(x7)
        d3 = torch.cat((x5, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x3, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Up1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d0 = self.Up_conv1(d1)
        return d0
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.device = args[0]
        return self



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Contourletfusion(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(Contourletfusion, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
        # self.cnnn = ContourletCNN()
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

# =============================================================================

# =============================================================================
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.detailFeature = DetailFeatureExtraction()
        self.ccnn = ContourletCNN()
        self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
          
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        base_feature = self.baseFeature(out_enc_level1)
        #detail_feature = self.detailFeature(out_enc_level1)
        
        # ccnn_inp = kornia.resize(inp_img,(224,224))
        self.ccnn.to('cuda')
        detail_input = self.Conv_1x1(out_enc_level1)
        detail_feature = self.ccnn(detail_input)
        detail_feature = kornia.resize(detail_feature,base_feature.shape[2:])
        return base_feature, detail_feature, out_enc_level1

class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, inp_img, base_feature, detail_feature):
       
        # print(base_feature.shape, detail_feature.shape)
        
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1) # 从第二个维度开始拼接
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0
    
if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    modelE = Restormer_Encoder().cuda()
    modelD = Restormer_Decoder().cuda()
