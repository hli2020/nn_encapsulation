import torch.nn as nn
from layers.models.cifar.resnet import BasicBlock, Bottleneck
from layers.cap_layer import CapLayer, CapConv, CapFC
from object_detection.utils.util import weights_init
import time
import torch


class CapsNet(nn.Module):
    """
    Capsule network.
    """
    def __init__(self, opts, num_classes=100):
        super(CapsNet, self).__init__()

        self.use_imagenet = True if opts.dataset == 'tiny_imagenet' else False
        self.cap_model = opts.cap_model
        self.use_multiple = opts.use_multiple
        input_ch = 1 if opts.dataset == 'fmnist' else 3
        self.measure_time = opts.measure_time
        self.cap_N = opts.cap_N

        if self.cap_model == 'v_base':
            # resnet baseline
            if hasattr(opts, 'depth'):
                depth = opts.depth
            else:
                depth = 20  # default value
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            self.inplanes = 16
            n = (depth - 2) / 6
            block = Bottleneck if depth >= 44 else BasicBlock

            if opts.dataset == 'tiny_imagenet':
                if opts.bigger_input:
                    stride_num, fc_num, pool_stride, pool_kernel = 2, 64*4*4, 7, 7
                else:
                    stride_num, fc_num, pool_stride, pool_kernel = 2, 64*2*2, 8, 8
            else:
                stride_num, fc_num, pool_stride, pool_kernel = 1, 64, 1, 8

            self.conv1 = nn.Conv2d(input_ch, 16, kernel_size=3, padding=1, bias=False, stride=stride_num)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2)
            self.avgpool = nn.AvgPool2d(pool_kernel, stride=pool_stride)
            self.fc = nn.Linear(fc_num, num_classes)

        elif self.cap_model == 'v0':
            # original capsule idea in the paper

            # first conv
            self.tranfer_conv = nn.Conv2d(input_ch, opts.pre_ch_num, kernel_size=9, padding=1, stride=2)  # 256x13x13
            self.tranfer_bn = nn.InstanceNorm2d(opts.pre_ch_num, affine=True) \
                if opts.use_instanceBN else nn.BatchNorm2d(opts.pre_ch_num)
            self.tranfer_relu = nn.ReLU(True)

            # second conv
            factor = 1 if opts.comp_cap else 8
            send_to_cap_ch_num = opts.primary_cap_num * factor
            self.tranfer_conv1 = nn.Conv2d(opts.pre_ch_num, send_to_cap_ch_num, kernel_size=3, stride=2)  # (say256)x6x6
            self.tranfer_bn1 = nn.InstanceNorm2d(send_to_cap_ch_num, affine=True) \
                if opts.use_instanceBN else nn.BatchNorm2d(send_to_cap_ch_num)
            self.tranfer_relu1 = nn.ReLU(True)

            # needed for large spatial input on imagenet
            if opts.bigger_input:
                self.max_pool = nn.MaxPool2d(9, stride=9)  # input 224, before cap, 54 x 54
            else:
                self.max_pool = nn.MaxPool2d(5, stride=5)  # input 128, before cap, 30 x 30
            # capsLayer
            self.cap_layer = CapLayer(opts, num_in_caps=opts.primary_cap_num*6*6, num_out_caps=num_classes,
                                      out_dim=16, num_shared=opts.primary_cap_num, in_dim=8)

        elif self.cap_model[0:2] == 'v1':

            self.layer1 = nn.Sequential(*[
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True)
            ])  # 32 spatial output

            self.cap1_conv = CapConv(ch_num=32*1, ch_out=32*2, groups=32)
            self.cap1_conv_sub = CapConv(ch_num=32*2, groups=32, N=self.cap_N)
            # 32 spatial output

            self.cap2_conv = CapConv(ch_num=32*2, ch_out=32*4, groups=32, kernel_size=3, stride=2, pad=1)
            self.cap2_conv_sub = CapConv(ch_num=32*4, groups=32, N=self.cap_N)
            # 16 spatial output

            self.cap3_conv = CapConv(ch_num=32*4, ch_out=32*8, groups=32, kernel_size=3, stride=2, pad=1)
            self.cap3_conv_sub = CapConv(ch_num=32*8, groups=32, N=self.cap_N)
            # 8 spatial output

            if self.cap_model == 'v1_1':
                self.final_cls = CapLayer(
                    opts, num_in_caps=opts.primary_cap_num*8*8, num_out_caps=num_classes,
                    out_dim=16, in_dim=8, num_shared=opts.primary_cap_num)

            elif self.cap_model == 'v1_2':
                # increase cap_num and downsize spatial capsules
                self.cap4_conv = CapConv(ch_num=32*8, ch_out=32*16, groups=32, kernel_size=3, stride=2, pad=1)
                # output: bs, 32*16, 4, 4
                self.final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=num_classes,
                                       cap_dim=16, fc_manner=opts.fc_manner)

        # init the network
        for m in self.modules():
            weights_init(m)

    def forward(self, x, target=None, curr_iter=0, vis=None):
        stats, output, start = [], [], []

        if self.measure_time:
            torch.cuda.synchronize()
            start = time.perf_counter()

        if self.cap_model == 'v_base':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)                  # 16 x 32 x 32
            x = self.layer2(x)                  # 32 x 16 x 16
            x = self.layer3(x)                  # 64(for depth=20) x 8 x 8
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            output = self.fc(x)

        elif self.cap_model == 'v0':
            x = self.tranfer_conv(x)
            x = self.tranfer_bn(x)
            x = self.tranfer_relu(x)
            x = self.tranfer_conv1(x)
            x = self.tranfer_bn1(x)
            x = self.tranfer_relu1(x)
            if self.measure_time:
                torch.cuda.synchronize()
                print('the whole previous conv time: {:.4f}'.format(time.perf_counter() - start))
                start = time.perf_counter()
            if self.use_imagenet:
                x = self.max_pool(x)
            output, stats = self.cap_layer(x, target, curr_iter, vis)
            if self.measure_time:
                torch.cuda.synchronize()
                print('last cap total time: {:.4f}'.format(time.perf_counter() - start))

        elif self.cap_model == 'v1_1' or self.cap_model == 'v1_2':

            x = self.layer1(x)
            x = self.cap1_conv(x)
            x = self.cap1_conv_sub(x)
            x = self.cap2_conv(x)
            x = self.cap2_conv_sub(x)
            x = self.cap3_conv(x)
            x = self.cap3_conv_sub(x)

            if self.cap_model == 'v1_1':
                output, _ = self.final_cls(x)

            elif self.cap_model == 'v1_2':
                x = self.cap4_conv(x)
                output = self.final_cls(x)
        else:
            raise NameError('Unknown structure or capsule model type.')

        return output, stats

    def _make_layer(self, block, planes, blocks, stride=1):
        """make resnet sub-layers"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, int(blocks)):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

