import math
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from collections.abc import Iterable
pi = math.pi


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, real_ratio,out_channels,flag, kernel_size, in_channels=1, sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):
        super(SincConv, self).__init__()
        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.flag = flag
        self.real_ratio = real_ratio
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        # initialize filterbanks using Mel scale
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        # print("1、f:",f,f.shape)
        fmel = self.to_mel(f)  # Hz to mel conversion
        # print("2、fmel:",fmel)
        fmelmax = np.max(fmel)
        # print("fmelmax:",fmelmax)
        fmelmin = np.min(fmel)
        # print("fmelmin:", fmelmin)
        # ----------------------------------------------------------------------
        if flag==1:
            filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)  # 0~2840，20
        else:
            list2 = np.array(
                [fmelmin, fmelmax / 8, fmelmax / 4, fmelmax * 3 / 8, fmelmax / 2, fmelmax * 5 / 8, fmelmax * 6 / 8,
                 fmelmax * 7 / 8, fmelmax])
            list3 = [0]
            for i in range(8):
                # list3 = np.concatenate((list3, np.linspace(list2[i], list2[i + 1], real_ratio[i] + 1)))
                if self.real_ratio[i] == 0:
                    continue
                elif self.real_ratio[i] == 1:
                    list3.extend([list2[i], list2[i + 1]])
                else:
                    list3.extend(np.linspace(list2[i], list2[i + 1], self.real_ratio[i] + 1))
            list3.append(fmelmax)
            filbandwidthsmel = np.unique(list3)
        # -------------------------------------------------------------------------------------
        # print("3、filbandwidthsmel:", filbandwidthsmel,filbandwidthsmel.shape)
        filbandwidthsf = self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        # print("4、filbandwidthsf:", filbandwidthsf,filbandwidthsf.shape)
        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)  # [-512,513,1)
        # -512,512
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp / self.sample_rate)  #
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)
        band_pass_filter = self.band_pass.to(self.device)
        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)
        y = F.conv1d(x, self.filters, stride=self.stride,
                     padding=self.padding, dilation=self.dilation,
                     bias=None, groups=1)
        return y


class Residual_block_one(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block_one, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)
        else:
            self.downsample = False
        # self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        # out = self.mp(out)
        return out


class Residual_block_2(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block_2, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=25)

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(in_channels=25,
                               out_channels=128,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.conv2 = nn.Conv1d(in_channels=128,
                               out_channels=128,
                               padding=1,
                               kernel_size=3,
                               stride=1)
        self.conv_downsample = nn.Conv1d(in_channels=25,
                                         out_channels=128,
                                         padding=0,
                                         kernel_size=1,
                                         stride=1)
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


# ------------------------------------CBAM-----------------------------------------
class CBAMLayer(nn.Module):  # CBAM
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # self.device = device
        # channel attention
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channels, channels // reduction, 1, bias=False),

            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = x.to(self.device)
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


# -------------------------ResNet-34---------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,is_last=False, **kwargs):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)

        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out=F(X)+X
        out += identity
        # -----------------------
        preact = out
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):

        super(ResNet, self).__init__()
        self.include_top = include_top  # 4
        self.in_channel = 64

        self.groups = groups  # 32
        self.width_per_group = width_per_group  # 4

        self.conv1 = nn.Conv1d(25, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.CBAM_64 = CBAMLayer( channels=64)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.CBAM_128 = CBAMLayer( channels=128)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.CBAM_256 = CBAMLayer(channels=256)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.CBAM_512 = CBAMLayer(channels=512)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # -> 2
        self.logsoftmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:  # expansion:4
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group,
                            is_last=(block_num == 1)
                            ))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                is_last=(_ == block_num - 1)
                                ))


        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f0 = x

        x, f1_pre = self.layer1(x)
        f1 = x
        x = x.unsqueeze(2)
        x = self.CBAM_64(x)
        x = x.squeeze(2)

        x, f2_pre = self.layer2(x)
        f2 = x
        x = x.unsqueeze(2)
        x = self.CBAM_128(x)
        x = x.squeeze(2)

        x, f3_pre = self.layer3(x)
        f3 = x
        x = x.unsqueeze(2)
        x = self.CBAM_256(x)
        x = x.squeeze(2)

        x, f4_pre = self.layer4(x)
        f4 = x
        x = x.unsqueeze(2)
        x = self.CBAM_512(x)
        x = x.squeeze(2)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f5 = x
        x = self.fc(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64, is_last=False):  # groups:32 width_per_group:4
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv1d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm1d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion)
        # -----------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        # -----------------------
        preact = out
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out
        # ------------------------


class ResNeXt(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):  # 4
        super(ResNeXt, self).__init__()
        self.include_top = include_top  # 4
        self.in_channel = 64

        self.groups = groups  # 32
        self.width_per_group = width_per_group  # 4

        self.conv1 = nn.Conv1d(25, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.CBAM_64 = CBAMLayer( channels=256)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.CBAM_128 = CBAMLayer( channels=512)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.CBAM_256 = CBAMLayer(channels=1024)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.CBAM_512 = CBAMLayer(channels=2048)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # -> 2
        self.logsoftmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:  # expansion:4
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group,
                            is_last=(block_num == 1)
                            ))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                is_last=(_ == block_num - 1)
                                ))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f0 = x

        x, f1_pre = self.layer1(x)
        f1 = x
        x = x.unsqueeze(2)
        x = self.CBAM_64(x)
        x = x.squeeze(2)

        x, f2_pre = self.layer2(x)
        f2 = x
        x = x.unsqueeze(2)
        x = self.CBAM_128(x)
        x = x.squeeze(2)

        x, f3_pre = self.layer3(x)
        f3 = x
        x = x.unsqueeze(2)
        x = self.CBAM_256(x)
        x = x.squeeze(2)

        x, f4_pre = self.layer4(x)
        f4 = x
        x = x.unsqueeze(2)
        x = self.CBAM_512(x)
        x = x.squeeze(2)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f5 = x
        x = self.fc(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x


class RawNet1(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet1, self).__init__()
        """
        d_args:
            model:
              nb_samp: 64600
              first_conv: 2048   # no. of filter coefficients  1024
              in_channels: 1
              filts: [20, [20, 20], [20, 128], [128, 128]] # no. of filters channel in residual blocks
              blocks: [2, 4]
              nb_fc_node: 1024
              gru_node: 1024
              nb_gru_layer: 3
              nb_classes: 2
        """
        groups = 32
        width_per_group = 4
        self.device = device  # gpu
        self.reanext50 = ResNeXt(Bottleneck, blocks_num=[3, 4, 6, 3], num_classes=2, include_top=True,groups=groups,
                                 width_per_group=width_per_group)
        # ---------------------------------------------------------
        self.first_bn = nn.BatchNorm1d(num_features=d_args['filts'][0])
        self.selu = nn.SELU(inplace=True)
        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.block0 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][1], first=True))
        self.block1 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_attention0 = self._make_attention_fc(in_features=d_args['filts'][1][-1],
                                                     l_out_features=d_args['filts'][1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features=d_args['filts'][1][-1],
                                                     l_out_features=d_args['filts'][1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention3 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention4 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention5 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features=d_args['filts'][2][-1])
        self.gru = nn.GRU(input_size=d_args['filts'][2][-1],
                          hidden_size=d_args['gru_node'],
                          num_layers=d_args['nb_gru_layer'],
                          batch_first=True)
        self.fc1_gru = nn.Linear(in_features=d_args['gru_node'],
                                 out_features=d_args['nb_fc_node'])

        self.fc2_gru = nn.Linear(in_features=d_args['nb_fc_node'],
                                 out_features=d_args['nb_classes'], bias=True)

    def forward(self, x, y=None):  # x: [128,64600],batch_size
        max_indexs = []
        intervals = []   #
        # ----------------------------------------F-ratio------------------------------
        x_numpy = x.detach().cpu().numpy()
        y_numpy = y.detach().cpu().numpy()
        flag = 0
        # freq_ranges = [np.linspace(i, i + 511, 512) for i in range(1, 4096, 512)]
        real_highest_frequency = 8000
        real_lowest_frequency = 0
        mel_highest_frequency = math.trunc(2595 * np.log10(1 + (real_highest_frequency) / 700))
        mel_lowest_frequency = math.trunc(2595 * np.log10(1 + (real_lowest_frequency) / 700))
        mel_mean_value = math.trunc(mel_highest_frequency/8)
        mel_ranges = [np.linspace(i, i + mel_mean_value - 1, mel_mean_value) for i in range(mel_lowest_frequency+1, mel_highest_frequency, mel_mean_value)]
        real_ranges = [700 * (10 ** (i / 2595) - 1) for i in mel_ranges]
        real_ranges_numpy = np.array(real_ranges)
        # ------------------------------------------------------------
        indexs = 4096*real_ranges_numpy/8000
        max_indexs.append(np.trunc([max(index) for index in indexs]))
        max_indexs = max_indexs[0]
        max_indexs = np.insert(max_indexs, 0, 0)
        max_indexs = max_indexs.astype(int)
        for i in range(len(max_indexs)-1):
            strat = max_indexs[i] if i == 0 else max_indexs[i]+1
            end = max_indexs[i+1]
            interval = list(range(strat, end+1))
            intervals.append(interval)

        x_spoof_magnitude = []
        x_bonafide_magnitude = []
        band_ratio = []
        # -----------------------------
        for i, (x_num, y_num) in enumerate(zip(x_numpy, y_numpy)):
            if y_num == 0:
                fft_result = np.fft.fft(x_num)
                freqs = np.fft.fftfreq(len(fft_result), 1 / 16000)
                positive_freqs = freqs[:len(freqs) // 2]
                magnitude = np.abs(fft_result[:len(freqs) // 2])  # abs
                x_spoof_magnitude.append(magnitude)
            elif y_num == 1:
                fft_result = np.fft.fft(x_num)
                freqs = np.fft.fftfreq(len(fft_result), 1 / 16000)
                positive_freqs = freqs[:len(freqs) // 2]
                magnitude = np.abs(fft_result[:len(freqs) // 2])  # abs
                x_bonafide_magnitude.append(magnitude)
            else:
                print("Don't exist！")
                exit(0)
        del x_numpy, y_numpy, fft_result, freqs, positive_freqs,magnitude
        x_spoof_magnitude_numpy = np.array(x_spoof_magnitude)
        x_bonafide_magnitude_numpy = np.array(x_bonafide_magnitude)
        del x_spoof_magnitude, x_bonafide_magnitude
        # --------------------------------------------
        df_fenzi = []
        for freq_range in intervals:
            df_fenzi.append(sum([self.fratio(num=i,
                                             magnitude_spoof=x_spoof_magnitude_numpy,
                                             magnitude_bonafide=x_bonafide_magnitude_numpy) for i in freq_range]))
        df_fenmu = sum(df_fenzi)
        del x_spoof_magnitude_numpy, \
            x_bonafide_magnitude_numpy
        if df_fenmu == 0:  #
            band_ratio = [0] * 8
            flag = 1
        else:
            df1 = df_fenzi[0] / df_fenmu if not math.isnan(df_fenzi[0]) else 0
            df2 = df_fenzi[1] / df_fenmu if not math.isnan(df_fenzi[1]) else 0
            df3 = df_fenzi[2] / df_fenmu if not math.isnan(df_fenzi[2]) else 0
            df4 = df_fenzi[3] / df_fenmu if not math.isnan(df_fenzi[3]) else 0
            df5 = df_fenzi[4] / df_fenmu if not math.isnan(df_fenzi[4]) else 0
            df6 = df_fenzi[5] / df_fenmu if not math.isnan(df_fenzi[5]) else 0
            df7 = df_fenzi[6] / df_fenmu if not math.isnan(df_fenzi[6]) else 0
            df8 = df_fenzi[7] / df_fenmu if not math.isnan(df_fenzi[7]) else 0
            band_ratio = [round(i * 25) if not math.isnan(i) else 0 for i in [df1, df2, df3, df4, df5, df6, df7, df8]]
            band_ratio = np.array(band_ratio)
        # -----------------------------
        while sum(band_ratio) != 25:
            if sum(band_ratio) < 25:
                max_value = np.argmax(band_ratio)
                band_ratio[max_value] += (25 - sum(band_ratio))
            elif sum(band_ratio) > 25:
                min_value = self.find_min_not_zero_or_one(lst=band_ratio)
                band_ratio[min_value] -= 1
        while self.find_min(band_ratio) == 0:
            suoyin_min = np.argmin(band_ratio)
            suoyin_max = np.argmax(band_ratio)
            band_ratio[suoyin_min] += 1
            band_ratio[suoyin_max] -= 1
        # -----------Sinc------------------------------
        self.Sinc_conv = SincConv(device=self.device, out_channels=25, kernel_size=1024,
                                  in_channels=1, real_ratio=band_ratio, flag=flag)
        del band_ratio
        # -----------------------------------------------------------------------------------------------------------
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)
        x = self.Sinc_conv(x.to(self.device))
        # print("x.shape:",x.shape)

        # ---------------Log---------------
        x = torch.log(torch.abs(x)+1e-5)
        # print("x.shape:",x.shape)
        x = F.max_pool1d(torch.abs(x), 3)
        # print("x:",x.shape)
        x = self.first_bn(x)
        x = self.selu(x)

        # ---------------------------------teacher model-------------------------
        feat, x = self.reanext50(x, is_feat=True, preact=True)
        x = self.logsoftmax(x)
        return feat, x
        # -----------------------------------------------------------------------

    # ------------fratio----------
    def fratio(self, num, magnitude_spoof, magnitude_bonafide):
        M = 2
        Num_spoof = magnitude_spoof.shape[0]
        Num_bonafide = magnitude_bonafide.shape[0]

        def calculate_miu(magnitude, num):
            return self.miu_i(mag=magnitude, num=num)

        def calculate_diff_squared(magnitude, miu, num):
            return sum((magnitude[j][num] - miu) ** 2 for j in range(magnitude.shape[0]))

        M_2_fenzi = 0
        M_2_fenmu = 0

        for i in range(M):
            if i == 0 and Num_spoof > 0:
                miu_spoof = self.miu(mag_spoof=magnitude_spoof, mag_bonafide=magnitude_bonafide, num=num)
                M_2_spoof_bonafide = (calculate_miu(magnitude_spoof, num) - miu_spoof) ** 2
                M_2_spoof_bonafide_fenmu = calculate_diff_squared(magnitude_spoof, calculate_miu(magnitude_spoof, num),
                                                                  num)
            elif i == 1 and Num_bonafide > 0:
                miu_bonafide = self.miu(mag_spoof=magnitude_spoof, mag_bonafide=magnitude_bonafide, num=num)
                M_2_spoof_bonafide = (calculate_miu(magnitude_bonafide, num) - miu_bonafide) ** 2
                M_2_spoof_bonafide_fenmu = calculate_diff_squared(magnitude_bonafide,
                                                                  calculate_miu(magnitude_bonafide, num), num)
            else:
                continue
            M_2_fenzi += M_2_spoof_bonafide
            M_2_fenmu = M_2_fenmu + M_2_spoof_bonafide_fenmu
        if M_2_fenmu == 0:
            return 0

        return M_2_fenzi / (M_2_fenmu * (Num_spoof + Num_bonafide))

    def miu_i(self,mag,num):  # mag:[27,4096]
        mean = np.mean(mag[:, num])
        return mean

    def miu(self,mag_spoof,mag_bonafide,num):
        all_miu = 0
        for i in mag_spoof:
            all_miu = all_miu + i[num]
        for j in mag_bonafide:
            all_miu = all_miu + j[num]
        # all_miu = sum(mag_spoof) + sum(mag_bonafide)
        count_miu = mag_spoof.shape[0] + mag_bonafide.shape[0]
        mean = all_miu/count_miu
        return mean

    def find_min_not_zero_or_one(self,lst):
        min_value = float('inf')
        min_index = -1
        for index, value in enumerate(lst):
            if value > 1 and value < min_value:
                min_value = value
                min_index = index
        return min_index

    def find_min(self,arr):

        min_value = arr[0]  # Assume the first element is the minimum initially

        for num in arr:
            if num < min_value:
                min_value = num

        return min_value

    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []

        l_fc.append(nn.Linear(in_features=in_features,
                              out_features=l_out_features))
        return nn.Sequential(*l_fc)

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn=None):
        if print_fn == None: printfn = print
        model = self  #

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    if len(summary[m_key]["output_shape"]) != 0:
                        summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()

        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)


class RawNet2(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet2, self).__init__()
        self.device = device  # gpu
        # ---------------------------------------------------------
        self.first_bn = nn.BatchNorm1d(num_features=d_args['filts'][0])
        self.selu = nn.SELU(inplace=True)
        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.block0 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][1], first=True))
        self.block1 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][1]))
        self.block2 = nn.Sequential(Residual_block_2(nb_filts=d_args['filts'][2]))
        self.block3 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][3]))
        self.block4 = nn.Sequential(Residual_block_one(nb_filts=d_args['filts'][3]))
        self.block5 = nn.Sequential(Residual_block_one(nb_filts=d_args['filts'][3]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_attention0 = self._make_attention_fc(in_features=d_args['filts'][1][-1],
                                                        l_out_features=d_args['filts'][1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features=d_args['filts'][1][-1],
                                                        l_out_features=d_args['filts'][1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features=d_args['filts'][3][-1],
                                                        l_out_features=d_args['filts'][3][-1])
        self.fc_attention3 = self._make_attention_fc(in_features=d_args['filts'][3][-1],
                                                        l_out_features=d_args['filts'][3][-1])
        self.fc_attention4 = self._make_attention_fc(in_features=d_args['filts'][3][-1],
                                                        l_out_features=d_args['filts'][3][-1])
        self.fc_attention5 = self._make_attention_fc(in_features=d_args['filts'][3][-1],
                                                        l_out_features=d_args['filts'][3][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features=d_args['filts'][3][-1])
        self.gru = nn.GRU(input_size=d_args['filts'][3][-1],
                              hidden_size=d_args['gru_node'],
                              num_layers=d_args['nb_gru_layer'],
                              batch_first=True)
        self.fc1_gru = nn.Linear(in_features=d_args['gru_node'],
                                     # nn.Linear
                                     out_features=d_args['nb_fc_node'])
        self.fc2_gru = nn.Linear(in_features=d_args['nb_fc_node'],
                                     out_features=d_args['nb_classes'], bias=True)
        self.cbam25 = CBAMLayer(25)
        self.cbam128 = CBAMLayer(128)

    def forward(self, x, y=None):  # x: [128,64600],batch_size

        max_indexs = []
        intervals = []   #
        # ----------------------------------------F-ratio------------------------------------------------------------
        x_numpy = x.detach().cpu().numpy()
        y_numpy = y.detach().cpu().numpy()
        flag = 0
        # --------------------------------------------------
        real_highest_frequency = 8000
        real_lowest_frequency = 0
        mel_highest_frequency = math.trunc(2595 * np.log10(1 + (real_highest_frequency) / 700))
        mel_lowest_frequency = math.trunc(2595 * np.log10(1 + (real_lowest_frequency) / 700))
        mel_mean_value = math.trunc(mel_highest_frequency/8)  # 取整
        mel_ranges = [np.linspace(i, i + mel_mean_value - 1, mel_mean_value) for i in range(mel_lowest_frequency+1, mel_highest_frequency, mel_mean_value)]
        real_ranges = [700 * (10 ** (i / 2595) - 1) for i in mel_ranges]  #
        real_ranges_numpy = np.array(real_ranges)  #
        # ------------------------------------------------------------
        indexs = 4096*real_ranges_numpy/8000
        max_indexs.append(np.trunc([max(index) for index in indexs]))
        max_indexs = max_indexs[0]
        max_indexs = np.insert(max_indexs, 0, 0)
        max_indexs = max_indexs.astype(int)
        for i in range(len(max_indexs)-1):
            strat = max_indexs[i] if i == 0 else max_indexs[i]+1
            end = max_indexs[i+1]
            interval = list(range(strat, end+1))
            intervals.append(interval)

        x_spoof_magnitude = []
        x_bonafide_magnitude = []
        band_ratio = []
        # ------------------------------
        for i, (x_num, y_num) in enumerate(zip(x_numpy, y_numpy)):
            if y_num == 0:
                fft_result = np.fft.fft(x_num)  # fft
                freqs = np.fft.fftfreq(len(fft_result), 1 / 16000)  #
                positive_freqs = freqs[:len(freqs) // 2]
                magnitude = np.abs(fft_result[:len(freqs) // 2])  # abs
                x_spoof_magnitude.append(magnitude)
            elif y_num == 1:
                fft_result = np.fft.fft(x_num)
                freqs = np.fft.fftfreq(len(fft_result), 1 / 16000)  #
                positive_freqs = freqs[:len(freqs) // 2]
                magnitude = np.abs(fft_result[:len(freqs) // 2])  # abs
                x_bonafide_magnitude.append(magnitude)
            else:
                print("Don't exist！")
                exit(0)
        del x_numpy, y_numpy, fft_result, freqs, positive_freqs,magnitude
        x_spoof_magnitude_numpy = np.array(x_spoof_magnitude)  #
        x_bonafide_magnitude_numpy = np.array(x_bonafide_magnitude)
        # x_bonafide_positive_freqs_mel_numpy = np.array(x_bonafide_positive_freqs_mel)
        del x_spoof_magnitude, x_bonafide_magnitude
        # --------------------------------------------
        df_fenzi = []
        for freq_range in intervals:
            df_fenzi.append(sum([self.fratio(num=i,
                                             magnitude_spoof=x_spoof_magnitude_numpy,
                                             magnitude_bonafide=x_bonafide_magnitude_numpy) for i in freq_range]))
        df_fenmu = sum(df_fenzi)
        del x_spoof_magnitude_numpy, \
            x_bonafide_magnitude_numpy
        if df_fenmu == 0:
            band_ratio = [0] * 8
            flag = 1
        else:
            df1 = df_fenzi[0] / df_fenmu if not math.isnan(df_fenzi[0]) else 0
            df2 = df_fenzi[1] / df_fenmu if not math.isnan(df_fenzi[1]) else 0
            df3 = df_fenzi[2] / df_fenmu if not math.isnan(df_fenzi[2]) else 0
            df4 = df_fenzi[3] / df_fenmu if not math.isnan(df_fenzi[3]) else 0
            df5 = df_fenzi[4] / df_fenmu if not math.isnan(df_fenzi[4]) else 0
            df6 = df_fenzi[5] / df_fenmu if not math.isnan(df_fenzi[5]) else 0
            df7 = df_fenzi[6] / df_fenmu if not math.isnan(df_fenzi[6]) else 0
            df8 = df_fenzi[7] / df_fenmu if not math.isnan(df_fenzi[7]) else 0
            band_ratio = [round(i * 25) if not math.isnan(i) else 0 for i in [df1, df2, df3, df4, df5, df6, df7, df8]]
            band_ratio = np.array(band_ratio)
        # ---------------------------
        while sum(band_ratio) != 25:
            if sum(band_ratio) < 25:
                max_value = np.argmax(band_ratio)
                band_ratio[max_value] += (25 - sum(band_ratio))
            elif sum(band_ratio) > 25:
                min_value = self.find_min_not_zero_or_one(lst=band_ratio)
                band_ratio[min_value] -= 1
        while self.find_min(band_ratio) == 0:
            suoyin_min = np.argmin(band_ratio)
            suoyin_max = np.argmax(band_ratio)
            band_ratio[suoyin_min] += 1
            band_ratio[suoyin_max] -= 1
        # -----------Sinc------------------------------
        self.Sinc_conv = SincConv(device=self.device, out_channels=25, kernel_size=1024,
                                  in_channels=1, real_ratio=band_ratio, flag=flag)
        del band_ratio
        # -----------------------------------------------------------------------------------------------------------
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)
        x = self.Sinc_conv(x.to(self.device))
        # print("x.shape:",x.shape)

        # ---------------log---------------
        x = torch.log(torch.abs(x)+1e-5)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)
        f0 = x
# -------------------------layer1----------------------------------------------------------------
        x0 = self.block0(x)

        y0 = self.avgpool(x0).view(x0.size(0), -1)  # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)

        x = x.unsqueeze(2)
        x = self.cbam25(x)
        x = x.squeeze(2)

        x1 = f1_pre = self.block1(x)

        y1 = self.avgpool(x1).view(x1.size(0), -1)  # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1  # (batch, filter, time) x (batch, filter, 1)
        f1 = x
        x = x.unsqueeze(2)
        x = self.cbam25(x)
        x = x.squeeze(2)
# -------------------------------layer2--------------------------------------------------------
        x2 = f2_pre = self.block2(x)

        y2 = self.avgpool(x2).view(x2.size(0), -1)  # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2  # (batch, filter, time) x (batch, filter, 1)
        f2 = x
        x = x.unsqueeze(2)
        x = self.cbam128(x)
        x = x.squeeze(2)
# ---------------------------layer3------------------------------------------------------
        x3 = f3_pre = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1)  # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3  # (batch, filter, time) x (batch, filter, 1)
        f3 = x
        x = x.unsqueeze(2)
        x = self.cbam128(x)
        x = x.squeeze(2)
# ---------------------layer4---------------------------------------------------------
        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1)  # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4  # (batch, filter, time) x (batch, filter, 1)
        x = x.unsqueeze(2)
        x = self.cbam128(x)
        x = x.squeeze(2)

        x5 = f4_pre = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1)  # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5  # (batch, filter, time) x (batch, filter, 1)
        f4 = x
        x = x.unsqueeze(2)
        x = self.cbam128(x)
        x = x.squeeze(2)
# --------------------------------------------------------------------------------
        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)

        f5 = x

        x = self.fc2_gru(x)
        output = self.logsoftmax(x)
        feat = [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5]
        return feat, output
        # ---------------------------------------------------------------------------------------------------------

    # ----------------------
    def fratio(self, num, magnitude_spoof, magnitude_bonafide):
        M = 2
        Num_spoof = magnitude_spoof.shape[0]
        Num_bonafide = magnitude_bonafide.shape[0]

        def calculate_miu(magnitude, num):
            return self.miu_i(mag=magnitude, num=num)

        def calculate_diff_squared(magnitude, miu, num):
            return sum((magnitude[j][num] - miu) ** 2 for j in range(magnitude.shape[0]))

        M_2_fenzi = 0
        M_2_fenmu = 0

        for i in range(M):
            if i == 0 and Num_spoof > 0:
                miu_spoof = self.miu(mag_spoof=magnitude_spoof, mag_bonafide=magnitude_bonafide, num=num)
                M_2_spoof_bonafide = (calculate_miu(magnitude_spoof, num) - miu_spoof) ** 2
                M_2_spoof_bonafide_fenmu = calculate_diff_squared(magnitude_spoof, calculate_miu(magnitude_spoof, num),
                                                                  num)
            elif i == 1 and Num_bonafide > 0:
                miu_bonafide = self.miu(mag_spoof=magnitude_spoof, mag_bonafide=magnitude_bonafide, num=num)
                M_2_spoof_bonafide = (calculate_miu(magnitude_bonafide, num) - miu_bonafide) ** 2
                M_2_spoof_bonafide_fenmu = calculate_diff_squared(magnitude_bonafide,
                                                                  calculate_miu(magnitude_bonafide, num), num)
            else:
                continue
            M_2_fenzi += M_2_spoof_bonafide
            M_2_fenmu = M_2_fenmu + M_2_spoof_bonafide_fenmu
        if M_2_fenmu == 0:
            return 0

        return M_2_fenzi / (M_2_fenmu * (Num_spoof + Num_bonafide))

    def miu_i(self,mag,num):  # mag:[27,4096]
        mean = np.mean(mag[:, num])
        return mean

    def miu(self,mag_spoof,mag_bonafide,num):
        all_miu = 0
        for i in mag_spoof:
            all_miu = all_miu + i[num]
        for j in mag_bonafide:
            all_miu = all_miu + j[num]
        # all_miu = sum(mag_spoof) + sum(mag_bonafide)
        count_miu = mag_spoof.shape[0] + mag_bonafide.shape[0]
        mean = all_miu/count_miu
        return mean

    def find_min_not_zero_or_one(self,lst):
        min_value = float('inf')
        min_index = -1
        for index, value in enumerate(lst):
            if value > 1 and value < min_value:
                min_value = value
                min_index = index
        return min_index

    def find_min(self,arr):

        min_value = arr[0]  # Assume the first element is the minimum initially

        for num in arr:
            if num < min_value:
                min_value = num

        return min_value

    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []

        l_fc.append(nn.Linear(in_features=in_features,
                              out_features=l_out_features))
        return nn.Sequential(*l_fc)

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn=None):

        if print_fn == None: printfn = print
        model = self

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    if len(summary[m_key]["output_shape"]) != 0:
                        summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()

        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)
