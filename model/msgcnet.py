import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange






class MultiScaleBrige(nn.Module):
    def __init__(self, in_channels_x1, in_channels_x2, in_channels_x3,
                 reduced_channels=None, key_channels_ratio=1, head_count=2, value_channels=None):
        super().__init__()

        self.in_channels_x1 = in_channels_x1
        self.in_channels_x2 = in_channels_x2
        self.in_channels_x3 = in_channels_x3
        # self.in_channels_x4 = in_channels_x4
        self.reduced_channels = reduced_channels

        self.head_count = head_count

        # EfficientCrossAttention  (in_channels_x1, in_channels_x2, key_channels, head_count, value_channels)
        self.eff_crs_att12 = EfficientCrossAttention(in_channels_x1, in_channels_x2, in_channels_x2 // key_channels_ratio, head_count, in_channels_x2)
        self.eff_crs_att13 = EfficientCrossAttention(in_channels_x1, in_channels_x3, in_channels_x3 // key_channels_ratio, head_count, in_channels_x3)
        self.eff_crs_att21 = EfficientCrossAttention(in_channels_x2, in_channels_x1, in_channels_x1 // key_channels_ratio, head_count, in_channels_x1)
        self.eff_crs_att23 = EfficientCrossAttention(in_channels_x2, in_channels_x3, in_channels_x3 // key_channels_ratio, head_count, in_channels_x3)
        self.eff_crs_att31 = EfficientCrossAttention(in_channels_x3, in_channels_x1, in_channels_x1 // key_channels_ratio, head_count, in_channels_x1)
        self.eff_crs_att32 = EfficientCrossAttention(in_channels_x3, in_channels_x2, in_channels_x2 // key_channels_ratio, head_count, in_channels_x2)

        # key_channels_ratio = [1, 2, 4]
        # # key_channels_ratio = [2, 4, 8]
        # # key_channels_ratio = [4, 8, 16]
        # # key_channels_ratio = [8, 16, 32]
        # self.eff_crs_att12 = EfficientCrossAttention(in_channels_x1, in_channels_x2, in_channels_x2 // key_channels_ratio[1], head_count, in_channels_x2)
        # self.eff_crs_att13 = EfficientCrossAttention(in_channels_x1, in_channels_x3, in_channels_x3 // key_channels_ratio[2], head_count, in_channels_x3)
        # self.eff_crs_att21 = EfficientCrossAttention(in_channels_x2, in_channels_x1, in_channels_x1 // key_channels_ratio[0], head_count, in_channels_x1)
        # self.eff_crs_att23 = EfficientCrossAttention(in_channels_x2, in_channels_x3, in_channels_x3 // key_channels_ratio[2], head_count, in_channels_x3)
        # self.eff_crs_att31 = EfficientCrossAttention(in_channels_x3, in_channels_x1, in_channels_x1 // key_channels_ratio[0], head_count, in_channels_x1)
        # self.eff_crs_att32 = EfficientCrossAttention(in_channels_x3, in_channels_x2, in_channels_x2 // key_channels_ratio[1], head_count, in_channels_x2)


        self.conv_out1 = nn.Conv2d(in_channels_x1+in_channels_x2+in_channels_x3, in_channels_x1, 1, 1, 0)
        self.conv_out2 = nn.Conv2d(in_channels_x1+in_channels_x2+in_channels_x3, in_channels_x2, 1, 1, 0)
        self.conv_out3 = nn.Conv2d(in_channels_x1+in_channels_x2+in_channels_x3, in_channels_x3, 1, 1, 0)



    def forward(self, x1, x2, x3):
        y12 = self.eff_crs_att12(x1, x2)
        y13 = self.eff_crs_att13(x1, x3)
        y21 = self.eff_crs_att21(x2, x1)
        y23 = self.eff_crs_att23(x2, x3)
        y31 = self.eff_crs_att31(x3, x1)
        y32 = self.eff_crs_att32(x3, x2)

        out1 = self.conv_out1(torch.cat([x1, y12, y13], dim=1))
        out2 = self.conv_out2(torch.cat([x2, y21, y23], dim=1))
        out3 = self.conv_out3(torch.cat([x3, y31, y32], dim=1))

        return out1, out2, out3


class EfficientCrossAttention(nn.Module):

    def __init__(self, in_channels_x1, in_channels_x2, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels_x1 = in_channels_x1
        self.in_channels_x2 = in_channels_x2
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels_x2, key_channels, 1)
        self.queries = nn.Conv2d(in_channels_x1, key_channels, 1)
        self.values = nn.Conv2d(in_channels_x2, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels_x2, 1)

    def forward(self, x1, x2):
        n, c1, h1, w1 = x1.size()
        n, c2, h2, w2 = x2.size()
        keys = self.keys(x2).reshape((n, self.key_channels, h2 * w2))
        queries = self.queries(x1).reshape(n, self.key_channels, h1 * w1)
        values = self.values(x2).reshape((n, self.value_channels, h2 * w2))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h1, w1)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        # attention = reprojected_value + x1
        attention = reprojected_value

        return attention



class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )

class Mlp3(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features),
            nn.BatchNorm2d(hidden_features),
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv3x3(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowEfficientSelfAttention(nn.Module):


    def __init__(self,
                 in_channels,
                 att_dim=None,
                 key_channels_redution=2,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 with_pos=True,

                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d,
                 drop_path_rate=0.,
                 drop=0.,
                 ):
        super().__init__()
        self.in_channels = in_channels
        if att_dim is None:
            att_dim = in_channels
        self.att_dim = att_dim
        self.key_channels_reduction = key_channels_redution
        self.num_heads = num_heads
        head_dim = att_dim // self.num_heads
        self.channel_scale = (head_dim // key_channels_redution) ** -0.5
        self.qkv_bias = qkv_bias
        self.ws = window_size
        self.with_pos = with_pos



        if self.with_pos == True:
            self.pos = PA(in_channels)

        self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=att_dim // key_channels_redution, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=in_channels, out_channels=att_dim // key_channels_redution, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=in_channels, out_channels=att_dim, kernel_size=1)

        self.head_proj = nn.Conv2d(att_dim, att_dim, 1, 1)
        self.proj = SeparableConvBN(att_dim, att_dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm1 = norm_layer(att_dim)
        self.norm2 = norm_layer(att_dim)

        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.mlp = Mlp3(in_features=att_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps, 0, 0), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        # x = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0.)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        # print(Hp, Wp)

        if self.with_pos:
            x = self.pos(x)

        stage1_shorcut = x

        x = self.norm1(x)

        q = self.conv_query(x)
        q = rearrange(q, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=self.att_dim//self.key_channels_reduction//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        k = self.conv_key(x)
        k = rearrange(k, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=self.att_dim//self.key_channels_reduction//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        v = self.conv_value(x)
        v = rearrange(v, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=self.att_dim//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        q = nn.functional.softmax(q, dim=-1)
        k = nn.functional.softmax(k, dim=-2)
        context = k.permute(0, 1, 3, 2) @ v
        attn_spatial = q @ context

        dots_channel = (q.transpose(-2, -1) @ k).transpose(-2, -1) * self.channel_scale
        dots_channel_max = nn.functional.adaptive_max_pool2d(dots_channel, 1)
        dots_channel_avg = nn.functional.adaptive_avg_pool2d(dots_channel, 1)
        dots_channel = dots_channel_avg + dots_channel_max

        attn_sptial_channel = attn_spatial * dots_channel

        attn_sptial_channel = rearrange(attn_sptial_channel, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=self.att_dim//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)


        stage1 = attn_sptial_channel[:, :, :H, :W]
        stage1 = self.head_proj(stage1)

        stage1_shorcut = stage1_shorcut[:, :, :H, :W]
        stage1 = self.drop_path(stage1) + stage1_shorcut

        stage2_shortcut = stage1

        stage2 = self.norm2(stage1)

        stage2 = self.mlp(stage2)
        x = stage2_shortcut + self.drop_path(stage2)

        return x


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class WF2(nn.Module):
    def __init__(self, in_channels=512, out_channels=256, eps=1e-8):
        super(WF2, self).__init__()
        self.pre_conv = Conv(in_channels, out_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(out_channels, out_channels, kernel_size=3)

    def forward(self, x, res):
        x = self.pre_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x



class Scale_Aware(nn.Module):
    def __init__(self, in_channels):
        super(Scale_Aware, self).__init__()

        # self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels)])
        self.conv1x1 = nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x_l, x_h):
        feat = torch.cat([x_l, x_h], dim=1)
        # feat=feat_cat.detach()
        feat = self.relu(self.conv1x1(feat))
        feat = self.relu(self.conv3x3_1(feat))
        att = self.conv3x3_2(feat)
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)

        fusion_1_2 = att_1 * x_l + att_2 * x_h
        return fusion_1_2


class SegHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=2, scale_factor=8):
        super().__init__()
        self.conv3x3 = ConvBNReLU(in_channels, in_channels, kernel_size=3)
        self.con1x1 = Conv(in_channels, num_classes, kernel_size=1)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.con1x1(x)
        out = self.up(x)
        return out


class MSGCNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 # decode_channels=64,
                 num_classes=2,
                 backbone_name='resnet34',
                 backbone_pretrained=None):
        super().__init__()
        self.in_channels = in_channels
        # self.decode_channels = decode_channels
        self.num_classes = num_classes


        if backbone_name == 'resnet34' or 'resnet18' or 'resnet50' or 'resnet101':
            self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                              out_indices=(1, 2, 3, 4), pretrained=True)
            encoder_channels = self.backbone.feature_info.channels()
            base_c = encoder_channels[0]


        self.msb = MultiScaleBrige(encoder_channels[0], encoder_channels[1], encoder_channels[2], head_count=1, key_channels_ratio=4)

        self.pre_conv = ConvBN(encoder_channels[-1], encoder_channels[-1], kernel_size=1)


        self.ff1 = WindowEfficientSelfAttention(in_channels=encoder_channels[-1], key_channels_redution=4, num_heads=16, window_size=16, with_pos=True)
        self.ff2 = WindowEfficientSelfAttention(in_channels=encoder_channels[-2], key_channels_redution=4, num_heads=8, window_size=16, with_pos=True)
        self.ff3 = WindowEfficientSelfAttention(in_channels=encoder_channels[-3], key_channels_redution=4, num_heads=4, window_size=16, with_pos=True)
        self.ff4 = WindowEfficientSelfAttention(in_channels=encoder_channels[-4], key_channels_redution=4, num_heads=2, window_size=16, with_pos=True)

        self.weight_sum1 = WF2(in_channels=encoder_channels[-1], out_channels=encoder_channels[-2])
        self.weight_sum2 = WF2(in_channels=encoder_channels[-2], out_channels=encoder_channels[-3])
        self.weight_sum3 = WF2(in_channels=encoder_channels[-3], out_channels=encoder_channels[-4])

        self.reduction1 = nn.Sequential(nn.Conv2d(in_channels=encoder_channels[-1], out_channels=encoder_channels[0], kernel_size=1),
                                        nn.BatchNorm2d(encoder_channels[0]),
                                        nn.ReLU())
        self.reduction2 = nn.Sequential(nn.Conv2d(in_channels=encoder_channels[-2], out_channels=encoder_channels[0], kernel_size=1),
                                        nn.BatchNorm2d(encoder_channels[0]),
                                        nn.ReLU())
        self.reduction3 = nn.Sequential(nn.Conv2d(in_channels=encoder_channels[-3], out_channels=encoder_channels[0], kernel_size=1),
                                        nn.BatchNorm2d(encoder_channels[0]),
                                        nn.ReLU())
        self.reduction4 = nn.Sequential(nn.Conv2d(in_channels=encoder_channels[-4], out_channels=encoder_channels[0], kernel_size=1),
                                        nn.BatchNorm2d(encoder_channels[0]),
                                        nn.ReLU())

        self.sa1 = Scale_Aware(in_channels=encoder_channels[0])
        self.sa2 = Scale_Aware(in_channels=encoder_channels[0])
        self.sa0 = Scale_Aware(in_channels=encoder_channels[0])

        self.seg_head = SegHead(in_channels=base_c, num_classes=num_classes, scale_factor=4)

        self.init_weight()

    def forward(self, x):

        x1, x2, x3, x4 = self.backbone(x)

        x1_ms, x2_ms, x3_ms = self.msb(x1, x2, x3)

        # 保持通道数不变
        x4 = self.ff1(self.pre_conv(x4))

        x3 = self.weight_sum1(x4, x3_ms)
        x3 = self.ff2(x3)

        x2 = self.weight_sum2(x3, x2_ms)
        x2 = self.ff3(x2)

        x1 = self.weight_sum3(x2, x1_ms)
        x1 = self.ff4(x1)

        x4 = self.reduction1(x4)
        x3 = self.reduction2(x3)
        x2 = self.reduction3(x2)
        x1 = self.reduction4(x1)

        x4 = nn.functional.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x3 = self.sa0(x4, x3)

        x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = self.sa1(x3, x2)


        x2 = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = self.sa2(x2, x1)

        out = self.seg_head(x1)

        return out



    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



if __name__ == '__main__':
    x = torch.rand((2, 3, 512, 512))
    model = MSGCNet(in_channels=3, num_classes=6, backbone_name='resnet34')
    out = model(x)
    print(out.shape)

