
import torch 
import logging
import torchvision
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


from .sam import  ImageEncoderViT, LayerNorm2d 
logger = logging.getLogger(__name__)

from typing import Type 


BN_MOMENTUM = 0.1


class PROMPT_MLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))



class Attention_block(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class EPSABlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(EPSABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes )
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result


class Bottle_ASPP(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        bn_mom     = 0.1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1   = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.aspp  = ASPP(planes, planes)
        self.conv2 = conv1x1(planes, inplanes)
        self.bn2   = nn.BatchNorm2d(inplanes, momentum=bn_mom)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self,x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.aspp(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = out + identity

        return out




class CFFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP_1 = nn.ModuleList([PROMPT_MLP(embedding_dim=1024, mlp_dim=int(256), act = nn.GELU)] * 4)
        self.MLP_2 = nn.ModuleList([PROMPT_MLP(embedding_dim=1024, mlp_dim=int(256), act = nn.GELU)] * 4)
        self.necks = nn.ModuleList([nn.Sequential(
                            nn.Conv2d(
                                1024,
                                256,
                                kernel_size=1,
                                bias=False,
                            ),
                            LayerNorm2d(256),
                            nn.Conv2d(
                                256,
                                256,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            LayerNorm2d(256),
                        )] * 4)
        self.convs1 = nn.ModuleList([EPSABlock(256, 256)] * 4)
        
        self.convs2 = nn.ModuleList([EPSABlock(256, 256)] * 4)

        self.convs3 = nn.ModuleList([EPSABlock(256, 256)] * 4)

        self.marges = nn.Sequential(
                    nn.Conv2d(in_channels=256 * 4, out_channels=256, kernel_size=1, stride=1, padding=0),
                    Bottle_ASPP(256, 64),
                )
    def forward(self,input):
        lows = []
        for i in range(4):
            input[i] = input[i].permute(0, 2, 3, 1) + self.MLP_1[i](input[i].permute(0, 2, 3, 1))
            input[i] = input[i] + self.MLP_2[i](input[i])
            input[i] = self.necks[i](input[i].permute(0, 3, 1, 2))
            lows.append(input[i])
        
        outs_1 = []

        outs_1.append( self.convs1[0](lows[0] + lows[1] + lows[2]))
        outs_1.append( self.convs1[1](lows[0] + lows[1] + lows[3]))
        outs_1.append( self.convs1[2](lows[0] + lows[2] + lows[3]))
        outs_1.append( self.convs1[3](lows[1] + lows[2] + lows[3]))

        outs_2 = []

        outs_2.append( self.convs2[0](outs_1[0] + outs_1[1] + outs_1[2]))
        outs_2.append( self.convs2[1](outs_1[0] + outs_1[1] + outs_1[3]))
        outs_2.append( self.convs2[2](outs_1[0] + outs_1[2] + outs_1[3]))
        outs_2.append( self.convs2[3](outs_1[1] + outs_1[2] + outs_1[3]))

        outs_3 = []

        outs_3.append( self.convs3[0](outs_2[0] + outs_2[1] + outs_2[2]))
        outs_3.append( self.convs3[1](outs_2[0] + outs_2[1] + outs_2[3]))
        outs_3.append( self.convs3[2](outs_2[0] + outs_2[2] + outs_2[3]))
        outs_3.append( self.convs3[3](outs_2[1] + outs_2[2] + outs_2[3]))

        outs = self.marges(torch.cat(outs_3, dim=1))
        return outs




class SFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_for = nn.Sequential(
            nn.Conv2d(3 , 64, kernel_size=1, stride=2,  padding=0, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, stride=2,  padding=0, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, stride=2,  padding=0, bias=False),

            EPSABlock(64, 64),
            EPSABlock(64, 64), 
            EPSABlock(64, 64),
        )
        self.atts = Attention_block(256 , 64, 64)


    def forward(self,input, mean_feature):
        x0 = self.down_for(input)
        x0 = self.atts(mean_feature, x0)
        return x0


class CFFD(nn.Module):
    def __init__(self):
        super().__init__()
        self.cdffm = CFFM()
        self.sfe   = SFE()

        self.last_layer = nn.Sequential(
                            nn.Conv2d(in_channels=256 + 64, out_channels=256, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),

                            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0), 
                            )


    def forward(self,input, outs):
        outs = self.cdffm(outs)

        out = F.interpolate(outs, size=(128, 128), mode='bilinear', align_corners=True)

        s = self.sfe(input, out)
        
        x = self.last_layer(torch.cat((out, s), dim=1))

        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        return x


class SAM_CFFNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        encoder_mode = {
            'l':
                {'name':'sam adapter',    'img_size': 1024,     'mlp_ratio': 4,           'patch_size': 16,       'qkv_bias': True,      'use_rel_pos': True, 
                'window_size': 14,       'out_chans': 256,     'scale_factor': 32,       'input_type': 'fft',    'freq_nums': 0.25,     'prompt_type': 'highpass', 
                'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor',  'embed_dim': 1024, 
                'depth': 24,             'num_heads': 16,      'global_attn_indexes': [5, 11, 17, 23] }
                            }[args.encoder_mode]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
                            img_size=args.inp_size,
                            patch_size=encoder_mode['patch_size'],
                            in_chans=3,
                            embed_dim=encoder_mode['embed_dim'],
                            depth=encoder_mode['depth'],
                            num_heads=encoder_mode['num_heads'],
                            mlp_ratio=encoder_mode['mlp_ratio'],
                            out_chans=encoder_mode['out_chans'],
                            qkv_bias=encoder_mode['qkv_bias'],
                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                            act_layer=nn.GELU,
                            use_rel_pos=encoder_mode['use_rel_pos'],
                            rel_pos_zero_init=True,
                            window_size=encoder_mode['window_size'],
                            global_attn_indexes=encoder_mode['global_attn_indexes'],
                        )
        if args.sam_pretrained_weights != None:
            sam_model_list = torch.load(args.sam_pretrained_weights)
            image_encoder_submodules = {name[14:]: state for name, state in sam_model_list.items() if 'image_encoder' in name}
            self.image_encoder.load_state_dict(image_encoder_submodules, strict=True)
            print("SAM.image_encoder 权重加载成功！")

        # 冻结 encoder
        for k, p in self.image_encoder.named_parameters():
            p.requires_grad = False

        self.decoder = CFFD()


    def forward(self,input):
        input = F.interpolate(input, scale_factor=4, mode='bilinear')

        _, outs = self.image_encoder.forward_blk(input)
        
        x = self.decoder(input, outs)
        return x



import torch 
import logging
import torchvision
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


from .sam import  ImageEncoderViT, LayerNorm2d 
logger = logging.getLogger(__name__)

from typing import Type 


BN_MOMENTUM = 0.1


class PROMPT_MLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))



class Attention_block(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class EPSABlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(EPSABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes )
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result


class Bottle_ASPP(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        bn_mom     = 0.1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1   = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.aspp  = ASPP(planes, planes)
        self.conv2 = conv1x1(planes, inplanes)
        self.bn2   = nn.BatchNorm2d(inplanes, momentum=bn_mom)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self,x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.aspp(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = out + identity

        return out




class CFFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP_1 = nn.ModuleList([PROMPT_MLP(embedding_dim=1024, mlp_dim=int(256), act = nn.GELU)] * 4)
        self.MLP_2 = nn.ModuleList([PROMPT_MLP(embedding_dim=1024, mlp_dim=int(256), act = nn.GELU)] * 4)
        self.necks = nn.ModuleList([nn.Sequential(
                            nn.Conv2d(
                                1024,
                                256,
                                kernel_size=1,
                                bias=False,
                            ),
                            LayerNorm2d(256),
                            nn.Conv2d(
                                256,
                                256,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            LayerNorm2d(256),
                        )] * 4)
        self.convs1 = nn.ModuleList([EPSABlock(256, 256)] * 4)
        
        self.convs2 = nn.ModuleList([EPSABlock(256, 256)] * 4)

        self.convs3 = nn.ModuleList([EPSABlock(256, 256)] * 4)


    def forward(self,input):
        lows = []
        for i in range(4):
            input[i] = input[i].permute(0, 2, 3, 1) + self.MLP_1[i](input[i].permute(0, 2, 3, 1))
            input[i] = input[i] + self.MLP_2[i](input[i])
            input[i] = self.necks[i](input[i].permute(0, 3, 1, 2))
            lows.append(input[i])
        
        outs_1 = []

        outs_1.append( self.convs1[0](lows[0] + lows[1] + lows[2]))
        outs_1.append( self.convs1[1](lows[0] + lows[1] + lows[3]))
        outs_1.append( self.convs1[2](lows[0] + lows[2] + lows[3]))
        outs_1.append( self.convs1[3](lows[1] + lows[2] + lows[3]))

        outs_2 = []

        outs_2.append( self.convs2[0](outs_1[0] + outs_1[1] + outs_1[2]))
        outs_2.append( self.convs2[1](outs_1[0] + outs_1[1] + outs_1[3]))
        outs_2.append( self.convs2[2](outs_1[0] + outs_1[2] + outs_1[3]))
        outs_2.append( self.convs2[3](outs_1[1] + outs_1[2] + outs_1[3]))

        outs_3 = []

        outs_3.append( self.convs3[0](outs_2[0] + outs_2[1] + outs_2[2]))
        outs_3.append( self.convs3[1](outs_2[0] + outs_2[1] + outs_2[3]))
        outs_3.append( self.convs3[2](outs_2[0] + outs_2[2] + outs_2[3]))
        outs_3.append( self.convs3[3](outs_2[1] + outs_2[2] + outs_2[3]))

        
        return outs_3




class SFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_for = nn.Sequential(
            nn.Conv2d(3 , 64, kernel_size=1, stride=2,  padding=0, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, stride=2,  padding=0, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, stride=2,  padding=0, bias=False),

            EPSABlock(64, 64),
            EPSABlock(64, 64), 
            EPSABlock(64, 64),
        )
        self.atts = Attention_block(256 , 64, 64)


    def forward(self,input, mean_feature):
        x0 = self.down_for(input)
        x0 = self.atts(mean_feature, x0)
        return x0


class CFFD(nn.Module):
    def __init__(self):
        super().__init__()
        self.cdffm = CFFM()
        self.sfe   = SFE()

        self.marges = nn.Sequential(
                    nn.Conv2d(in_channels=256 * 4, out_channels=256, kernel_size=1, stride=1, padding=0),
                    Bottle_ASPP(256, 64),
                )
        self.last_layer = nn.Sequential(
                            nn.Conv2d(in_channels=256 + 64, out_channels=256, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),

                            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0), 
                            )


    def forward(self,input, outs):
        outs = self.cdffm(outs)
        out  = self.marges(torch.cat(outs, dim=1))
        out  = F.interpolate(out, size=(128, 128), mode='bilinear', align_corners=True)

        s    = self.sfe(input, out)
        x    = self.last_layer(torch.cat((out, s), dim=1))
        x    = F.interpolate(x, scale_factor    = 2, mode='bilinear', align_corners=True)
        return x


class SAM_CFFNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        encoder_mode = {
            'l':
                {'name':'sam adapter',    'img_size': 1024,     'mlp_ratio': 4,           'patch_size': 16,       'qkv_bias': True,      'use_rel_pos': True, 
                'window_size': 14,       'out_chans': 256,     'scale_factor': 32,       'input_type': 'fft',    'freq_nums': 0.25,     'prompt_type': 'highpass', 
                'prompt_embed_dim': 256, 'tuning_stage': 1234, 'handcrafted_tune': True, 'embedding_tune': True, 'adaptor': 'adaptor',  'embed_dim': 1024, 
                'depth': 24,             'num_heads': 16,      'global_attn_indexes': [5, 11, 17, 23] }
                            }[args.encoder_mode]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
                            img_size=args.inp_size,
                            patch_size=encoder_mode['patch_size'],
                            in_chans=3,
                            embed_dim=encoder_mode['embed_dim'],
                            depth=encoder_mode['depth'],
                            num_heads=encoder_mode['num_heads'],
                            mlp_ratio=encoder_mode['mlp_ratio'],
                            out_chans=encoder_mode['out_chans'],
                            qkv_bias=encoder_mode['qkv_bias'],
                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                            act_layer=nn.GELU,
                            use_rel_pos=encoder_mode['use_rel_pos'],
                            rel_pos_zero_init=True,
                            window_size=encoder_mode['window_size'],
                            global_attn_indexes=encoder_mode['global_attn_indexes'],
                        )
        if args.sam_pretrained_weights != None:
            sam_model_list = torch.load(args.sam_pretrained_weights)
            image_encoder_submodules = {name[14:]: state for name, state in sam_model_list.items() if 'image_encoder' in name}
            self.image_encoder.load_state_dict(image_encoder_submodules, strict=True)
            print("SAM.image_encoder 权重加载成功！")

        # 冻结 encoder
        for k, p in self.image_encoder.named_parameters():
            p.requires_grad = False

        self.decoder = CFFD()


    def forward(self,input):
        input    = F.interpolate(input, size = (1024, 1024), mode='bilinear')
        _, outs = self.image_encoder.forward_blk(input)
        x = self.decoder(input, outs)
        return x


