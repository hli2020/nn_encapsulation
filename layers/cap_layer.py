import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
import math

EPS = np.finfo(float).eps

# ATTENTION!!!
# you may find some variables in this file are redundant/useless.
# Never mind; they might be disposed of in the future.
# Those are the preliminary investigation in the early stage of this project.


class CapLayer(nn.Module):
    """
        The original capsule implementation in the NIPS/ICLR paper;
        and detailed ablative analysis on it.
    """
    def __init__(self, opts, num_in_caps, num_out_caps,
                 out_dim, in_dim, num_shared, route, as_conv_output=False):
        super(CapLayer, self).__init__()

        self.as_conv_output = as_conv_output
        self.route = route

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num_shared = num_shared
        self.num_out_caps = num_out_caps
        self.num_in_caps = num_in_caps

        self.route_num = opts.route_num
        self.squash_manner = opts.squash_manner

        self.W = nn.Conv2d(num_shared*in_dim, num_shared*num_out_caps*out_dim,
                           kernel_size=1, stride=1, groups=num_shared, bias=True)

        if self.route == 'EM':
            self.beta_v = Parameter(torch.Tensor(self.num_out_caps))
            self.beta_a = Parameter(torch.Tensor(self.num_out_caps))
            nn.init.uniform(self.beta_v.data)
            nn.init.uniform(self.beta_a.data)
            self.fac_R_init = 1.
            self.fac_temperature = 1.
            self.fac_nor = 10.0
            self.E_step_norm = opts.E_step_norm

    def forward(self, input,
                target=None, curr_iter=None, draw_hist=None,
                activate=None):
        """
            Input:
                target, curr_iter, draw_hist are for debugging or stats collection purpose only;
                'activate' is the activation of previous layer: a_i (i being # of input capsules)
            return:
                v, stats, activation (a_j)
        """
        start, activation = [], []
        stats = []      # verbose; reserved for legacy

        bs, in_channels, h, w = input.size()
        pred_list = []

        # 1. affine mapping (get 'pred', \hat{v}_j|i)
        pred = self.W(input)
        # pred: bs, 5120, 6, 6
        # -> bs, 10, 16, 32x6x6 (view)
        spatial_size = pred.size(2)
        pred = pred.view(bs, self.num_shared, self.num_out_caps, self.out_dim,
                         spatial_size, spatial_size)
        pred = pred.permute(0, 2, 3, 1, 4, 5).contiguous()
        pred = pred.view(bs, pred.size(1), pred.size(2), -1)

        # 2. routing (get 'v')
        if self.route == 'dynamic':
            v = self.dynamic(bs, pred, pred_list)
        elif self.route == 'EM':
            v, activation = self.EM(bs, pred, activate)

        if self.as_conv_output:
            spatial_size_out = int(math.sqrt(self.num_out_caps/self.num_shared))
            v = v.view(v.size(0), -1, spatial_size_out, spatial_size_out, v.size(2))
            v = v.permute(0, 1, 4, 2, 3).contiguous()
            v = v.view(v.size(0), -1, v.size(3), v.size(4))
        return v, stats, activation

    def dynamic(self, bs, pred, pred_list):

        b = Variable(torch.zeros(bs, self.num_out_caps, self.num_in_caps),
                     requires_grad=False)

        for i in range(self.route_num):

            c = F.softmax(b, dim=2)   # dim=1 WON'T converge!!!!

            # TODO (mid): optimize time (0.0238)
            s = torch.matmul(pred, c.unsqueeze(3)).squeeze()

            # shape of v: bs, out_cap_num, out_cap_dim
            v = squash(s, self.squash_manner)

            # update b/c
            if i < self.route_num - 1:
                # TODO (mid): super inefficient (0.1557s)
                # delta_b = torch.matmul(_pred, v.unsqueeze(3)).squeeze()
                delta_b = torch.matmul(v.unsqueeze(2), pred).squeeze()
                b = torch.add(b, delta_b)

        return v

    def EM(self, bs, pred, activate):
        # V_ij is just pred (bs, 10, 16, 1152)
        # output capsule is mu_j (bs, 10, 16)
        R = Variable(torch.ones(self.num_in_caps, self.num_out_caps), requires_grad=False)
        R /= self.num_out_caps * self.fac_R_init
        R = R.repeat(bs, 1, 1)

        for i in range(self.route_num):
            activation, mu, std = self._M_step(R, activate, pred, i)
            if i < self.route_num-1:
                R = self._E_step(activation, mu, std, pred)

        return mu, activation

    def _M_step(self, R, activate, V, iter):
        # activate: bs, i
        # activation: bs, j
        if activate is not None:
            R = R*activate.view(activate.size(0), activate.size(1), 1)
        R_j = torch.sum(R, dim=1).unsqueeze(dim=2) + EPS  # bs, j, 1
        mu = torch.matmul(V, R.permute(0, 2, 1).unsqueeze(dim=3)).squeeze()
        mu = mu / R_j

        std_V = (V - mu.view(mu.size(0), mu.size(1), mu.size(2), 1))**2
        std = torch.matmul(std_V, R.permute(0, 2, 1).unsqueeze(dim=3)).squeeze()
        std = torch.sqrt(std / R_j)

        cost = torch.log(std + EPS) + self.beta_v.view(self.beta_v.size(0), 1)    # learned beta
        # cost = torch.log(std) + self.beta_v   # fixed beta
        cost *= R_j
        cost_sum = normalize_cap(torch.sum(cost, dim=2), self.fac_nor)
        cost_sum = self.beta_a - cost_sum
        temperature = (1.0+iter)/self.fac_temperature
        activation = F.sigmoid(temperature * cost_sum)

        return activation, mu, std

    def _E_step(self, activation, mu, std, V):

        prefix = 1 / (torch.sqrt(torch.prod(2*math.pi*(std**2), dim=2)) + EPS)
        deno = (2*(std**2)) + EPS
        x = (V - mu.unsqueeze(dim=3))**2 / deno.unsqueeze(dim=3)
        x = torch.sum(x, dim=2)
        if self.E_step_norm:
            x = normalize_cap(x, self.fac_nor)
        p = prefix.unsqueeze(dim=2) * torch.exp(-x)
        r = p * activation.unsqueeze(dim=2)
        R = r / (torch.sum(r, dim=1, keepdim=True) + EPS)

        # _check = np.isnan(R.data.cpu().numpy())
        # if _check.any():
        #     a = 1
        return R.permute(0, 2, 1)


class CapConv(nn.Module):
    """
        the basic capConv block
        if residual is true, use skip connection

            input
            |   | conv2d
            | + | (optional, skip connection)
            |   | BN-ReLU-Squash
            |   | conv2d
            | + | (optional, skip connection)
            |   | BN-ReLU-Squash
            |   | ... (N times)
            |   | conv2d
            | + | (optional, skip connection)
            |   | BN-ReLU-Squash
        """
    def __init__(self, ch_num, groups,
                 N=1, ch_out=-1,
                 kernel_size=(1,), stride=1, pad=(0,),
                 residual=False, manner='0',
                 layerwise_skip_connect=False,
                 use_capBN=False, skip_relu=False,
                 ):

        super(CapConv, self).__init__()
        ch_num_in = ch_num
        ch_num_out = ch_num if ch_out == -1 else ch_out

        layers = nn.ModuleList([])
        for i in range(N):
            layers.append(
                _make_core_conv(
                    manner=manner, use_capBN=False, skip_relu=False,
                    ch_num_in=ch_num_in, ch_num_out=ch_num_out,
                    kernel_size=kernel_size, stride=stride, groups=groups, pad=pad
                )
            )
            if use_capBN:
                layers.append(nn.BatchNorm3d(groups))
            else:
                layers.append(nn.BatchNorm2d(ch_num_out))
            if not skip_relu:
                layers.append(nn.ReLU(True))
            layers.append(conv_squash(groups))

        self.block = layers

        if residual:
            ls = list(range(N))
            interval = len(self.block)/N
            self.insert_input_ls = [int(i*interval) for i in ls]
            if not layerwise_skip_connect:
                self.insert_input_ls = [self.insert_input_ls[-1]]

            if ch_num_in != ch_num_out:
                "only applies in the main_conv"
                self.conv_adjust_blob_shape = \
                    nn.Conv2d(ch_num_in, ch_num_out,
                              kernel_size=3, padding=1, stride=stride)
        else:
            self.insert_input_ls = [-1]
        self.layerwise = layerwise_skip_connect
        self.groups = groups

    def forward(self, x, send_ot_output=False):
        """send_ot_output is to compute OT loss as input"""
        if not self.layerwise:
            x_original = x

        for i in range(len(self.block)):
            if i in self.insert_input_ls:
                assert isinstance(self.block[i], basic_conv) \
                       or isinstance(self.block[i], capConvRoute3)
                if self.layerwise:
                    # only for sub_conv whose N is greater than 1
                    x_input = x
                else:
                    # for both sub and main conv
                    x_input = x_original    # use the very first input
                x = self.block[i](x)
                if x_input.size(1) != x.size(1):
                    # only for main_conv
                    x_input = self.conv_adjust_blob_shape(x_input)
                x += x_input
            else:
                if isinstance(self.block[i], nn.BatchNorm3d):
                    x = x.view(x.size(0), self.groups, -1, x.size(2), x.size(3))
                x = self.block[i](x)
                if isinstance(self.block[i], nn.BatchNorm3d):
                    x = x.view(x.size(0), -1, x.size(3), x.size(4))

            if send_ot_output and i == (len(self.block)/2-1):
                ot_output = x

        if send_ot_output:
            return x, ot_output
        else:
            return x


class CapConv2(nn.Module):
    """
        Wrap up the CapConv layer with multiple sub layers into a module,
        possibly with skip connection.
    """
    def __init__(self, ch_in, ch_out, groups,
                 residual, iter_N,
                 no_downsample=False,               # for main_conv
                 wider_main_conv=False,             # for main_conv
                 layerwise_skip_connect=True,       # for sub_conv
                 manner='0',                        # for both
                 ot_choice=None,                    # send out intermediate outputs
                 use_capBN=False, skip_relu=False,
                 ):

        super(CapConv2, self).__init__()
        assert len(residual) == 2
        self.ot_choice = ot_choice

        # define main_conv
        "the main conv is for increasing cap_dim; larger ksize is needed"
        "by default it only iterates once (N=1)"
        main_ksize = (5, 3, 1) if wider_main_conv else (3,)
        main_pad = (2, 1, 0) if wider_main_conv else (1,)
        main_stride = 1 if no_downsample else 2
        self.main_conv = CapConv(ch_num=ch_in, ch_out=ch_out, groups=groups,
                                 kernel_size=main_ksize, stride=main_stride, pad=main_pad,
                                 residual=residual[0], manner=manner,
                                 use_capBN=use_capBN, skip_relu=skip_relu)

        # define sub_conv
        "the sub conv is for stabilizing the cap block at a fixed cap_dim; "
        "ksize is by default to be 1; stride/pad/ksize are set by default"
        # (old note) should be exactly the same as 'v1_3' in network.py
        self.sub_conv = CapConv(ch_num=ch_out, groups=groups, N=iter_N,
                                residual=residual[1], manner=manner,
                                layerwise_skip_connect=layerwise_skip_connect,
                                use_capBN=use_capBN, skip_relu=skip_relu)

    def forward(self, input):
        """
            within: between input and output of sub_conv
            within2: between input of main_conv and the HALF output of sub_conv
        """
        out_list = []
        if self.ot_choice == 'within2':
            out_list.append(input)      # as ground truth "y"
        out = self.main_conv(input)
        if self.ot_choice == 'within':
            out_list.append(out)        # as ground truth "y"

        if self.ot_choice == 'within':
            out = self.sub_conv(out)
            out_list.append(out)        # as latent variable "z"
        elif self.ot_choice == 'within2':
            out, ot_out = self.sub_conv(out, send_ot_output=True)
            out_list.append(ot_out)     # as latent variable "z"
        else:
            out = self.sub_conv(out)
        return out, out_list


class CapFC(nn.Module):
    """
        given an input 4-D blob, generate the FC output;
        this layer assumes the input capsule's dim is the same as that of the output's capsule
    """
    def __init__(self, in_cap_num, out_cap_num, cap_dim, fc_manner='default'):
        super(CapFC, self).__init__()
        self.in_cap_num = in_cap_num
        self.out_cap_num = out_cap_num
        self.cap_dim = cap_dim
        self.fc_manner = fc_manner
        if fc_manner == 'default':
            self.weight = Parameter(torch.Tensor(cap_dim, in_cap_num, out_cap_num))
            self.reset_parameters()
        elif fc_manner == 'fc':
            self.fc_layer = nn.Linear(self.in_cap_num*self.cap_dim,
                                      self.out_cap_num*self.cap_dim)
        elif fc_manner == 'paper':
            self.W1 = Parameter(torch.Tensor(cap_dim, cap_dim*out_cap_num))
            self.W2 = Parameter(torch.Tensor(out_cap_num, in_cap_num, 1))
            self.reset_parameters()

    def forward(self, x):
        if self.fc_manner == 'default':
            x = x.view(x.size(0), -1, self.cap_dim, x.size(2), x.size(3))
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = x.view(x.size(0), -1, self.cap_dim)
            input = x.permute(0, 2, 1).contiguous().unsqueeze(2)
            input = torch.matmul(input, self.weight).squeeze().permute(0, 2, 1).contiguous()
        elif self.fc_manner == 'fc':
            x = x.view(x.size(0), -1)
            input = self.fc_layer(x)
            input = input.view(input.size(0), self.out_cap_num, self.cap_dim)
        elif self.fc_manner == 'paper':
            x = x.view(x.size(0), -1, self.cap_dim, x.size(2), x.size(3))
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = x.view(x.size(0), -1, self.cap_dim)
            input = torch.matmul(x, self.W1)
            input = input.view(input.size(0), self.in_cap_num, self.cap_dim, self.out_cap_num)
            input = input.permute(0, 3, 2, 1).contiguous()
            input = torch.matmul(input, self.W2).squeeze()
        return squash(input)

    def reset_parameters(self):
        "init param here"
        if self.fc_manner == 'default':
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
        elif self.fc_manner == 'paper':
            stdv = 1. / math.sqrt(self.W1.size(1))
            self.W1.data.uniform_(-stdv, stdv)
            stdv = 1. / math.sqrt(self.W2.size(1))
            self.W2.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_cap_num=' + str(self.in_cap_num) \
            + ', out_cap_num=' + str(self.out_cap_num) \
            + ', cap_dim=' + str(self.cap_dim) \
            + ', fc_manner=' + self.fc_manner + ')'


# ============== Utilities below ==============
def normalize_cap(input, factor):
    """used in EM routing; input: bs, j, (i)"""
    min_input, _ = torch.min(input, dim=1, keepdim=True)
    max_input, _ = torch.max(input, dim=1, keepdim=True)
    output = \
        factor*(input - min_input) / (max_input - min_input + EPS) - factor/2

    return output


def _make_core_conv(
        ch_num_in, ch_num_out, kernel_size, stride, groups, pad,
        manner='0', use_capBN=False, skip_relu=False,):
    """
        used in convCap/convCap2 block
        kernel_size, pad, should be tuple type
    """
    conv_opt =[]
    if manner == '0':
        conv_opt = basic_conv(
            ch_num_in, ch_num_out, ksize=kernel_size,
            stride=stride, group=groups, pad=pad)

    elif manner == '3':
        conv_opt = capConvRoute3(
            ch_num_in, ch_num_out, ksize=kernel_size,
            stride=stride, group=groups, pad=pad,
            use_capBN=use_capBN, skip_relu=skip_relu,)
    elif manner == '4':
        conv_opt = capConvRoute4(
            ch_num_in, ch_num_out, ksize=kernel_size,
            stride=stride, group=groups, pad=pad,
            use_capBN=use_capBN, skip_relu=skip_relu,)

    return conv_opt


def squash(vec, manner='paper'):
    """given a 3-D (bs, num_cap, dim_cap) input, squash the capsules
    output has the same shape as input
    """
    coeff2, mean = [], []

    assert len(vec.size()) == 3
    norm = vec.norm(dim=2)

    if manner == 'paper':
        norm_squared = norm ** 2
        coeff = norm_squared / (1 + norm_squared)
        coeff2 = torch.unsqueeze((coeff/(norm+EPS)), dim=2)
        # coeff2: bs x num_cap x 1

    elif manner == 'sigmoid':
        try:
            mean = (norm.max() - norm.min()) / 2.0
        except RuntimeError:
            print(vec)
            print(norm)
        coeff = F.sigmoid(norm - mean)   # in-place bug
        coeff2 = torch.unsqueeze((coeff/norm), dim=2)

    return torch.mul(vec, coeff2)

# ============== Utilities END ==============


class capConvRoute3(nn.Module):
    def __init__(self,
                 ch_num_in, ch_num_out,
                 ksize, pad, stride, group,
                 use_capBN=False, skip_relu=False,):

        super(capConvRoute3, self).__init__()
        self.expand_factor = int(ch_num_out/group)  # just the out_cap_dim

        self.main_cap = nn.ModuleList([
            # could be wider
            basic_conv(ch_num_in, ch_num_out,
                       ksize=ksize, stride=stride, group=group, pad=pad),
            nn.BatchNorm2d(ch_num_out),
        ])
        if not skip_relu:
            self.main_cap.append(nn.ReLU())
        self.main_cap.append(conv_squash(group))

        # res_cap: ksize is larger than main_cap; NO GROUPING
        self.res_cap = nn.ModuleList([
            nn.Conv2d(ch_num_in, ch_num_out, kernel_size=ksize[0]+4,
                      stride=stride, padding=pad[0]+2),
            nn.BatchNorm2d(ch_num_out),
            nn.ReLU(),
            conv_squash(group)
        ])
        self.main_cap_coeff = nn.Conv2d(
            ch_num_out, group, kernel_size=3, stride=1, padding=1, groups=group)
        self.res_cap_coeff = nn.Conv2d(
            ch_num_out, group, kernel_size=3, stride=1, padding=1, groups=group)

    def forward(self, x):
        "consider the output of capsule combination as a simple convolution output"
        # co-efficients are communicated in a X-shape
        main_out, res_out = x, x
        for module in self.main_cap:
            main_out = module(main_out)
        res_coeff = self.main_cap_coeff(main_out)
        res_coeff = res_coeff.unsqueeze(dim=2).repeat(1, 1, self.expand_factor, 1, 1)
        res_coeff = res_coeff.view(res_coeff.size(0), -1, res_coeff.size(3), res_coeff.size(4))

        for module in self.res_cap:
            res_out = module(res_out)
        main_coeff = self.res_cap_coeff(res_out)
        # t = time.time()
        main_coeff = main_coeff.unsqueeze(dim=2).repeat(1, 1, self.expand_factor, 1, 1)
        main_coeff = main_coeff.view(main_coeff.size(0), -1, main_coeff.size(3), main_coeff.size(4))
        # print('time is {:.5f}'.format(time.time() - t))
        # t = time.time()
        # main_coeff = torch.cat(
        #     [main_coeff[:, i, :, :].unsqueeze(dim=1).repeat(1, self.expand_factor, 1, 1)
        #      for i in range(self.group)], dim=1)
        # print('time is {:.5f}'.format(time.time() - t))
        out = main_out * main_coeff + res_out * res_coeff
        return out


class capConvRoute4(capConvRoute3):
    def __init__(self,
                 ch_num_in, ch_num_out,
                 ksize, pad, stride, group,
                 use_capBN=False, skip_relu=False,):
        super(capConvRoute4, self).__init__(
                ch_num_in, ch_num_out, ksize, pad, stride, group,
                use_capBN=use_capBN, skip_relu=skip_relu)

    def forward(self, x):
        main_out, res_out = x, x
        for module in self.main_cap:
            main_out = module(main_out)
        main_coeff = self.main_cap_coeff(main_out)
        main_coeff = main_coeff.unsqueeze(dim=2).repeat(1, 1, self.expand_factor, 1, 1)
        main_coeff = main_coeff.view(main_coeff.size(0), -1, main_coeff.size(3), main_coeff.size(4))

        for module in self.res_cap:
            res_out = module(res_out)
        res_coeff = self.res_cap_coeff(res_out)
        # t = time.time()
        res_coeff = res_coeff.unsqueeze(dim=2).repeat(1, 1, self.expand_factor, 1, 1)
        res_coeff = res_coeff.view(res_coeff.size(0), -1, res_coeff.size(3), res_coeff.size(4))
        out = main_out * main_coeff + res_out * res_coeff
        return out


class basic_conv(nn.Module):
    """used in '_make_core_conv' method to build parallel convolutions"""
    def __init__(self,
                 ch_num_in, ch_num_out,
                 ksize, pad, stride, group):
        super(basic_conv, self).__init__()
        assert len(ksize) == len(pad)
        self.multi_N = len(ksize)

        layers = nn.ModuleList([])
        for i in range(self.multi_N):
            layers.append(nn.Conv2d(
                ch_num_in, ch_num_out, kernel_size=ksize[i],
                stride=stride, groups=group, padding=pad[i]))
        self.conv = layers

    def forward(self, input):
        for i in range(self.multi_N):
            if i == 0:
                out = self.conv[i](input)
            else:
                out += self.conv[i](input)
        return out


class conv_squash(nn.Module):
    def __init__(self, num_shared):
        super(conv_squash, self).__init__()
        self.num_shared = num_shared

    def forward(self, x):
        spatial_size = x.size(2)
        batch_size = x.size(0)
        cap_dim = int(x.size(1) / self.num_shared)
        x = x.view(batch_size, self.num_shared, cap_dim, spatial_size, spatial_size)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(batch_size, -1, cap_dim)
        x = squash(x)
        # revert back to original shape
        x = x.view(batch_size, self.num_shared, spatial_size, spatial_size, cap_dim)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(batch_size, -1, spatial_size, spatial_size)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'num_shared=' + str(self.num_shared) + ')'


class MarginLoss(nn.Module):
    def __init__(self, num_classes, pos=0.9, neg=0.1, lam=0.5):
        super(MarginLoss, self).__init__()
        self.num_classes = num_classes
        self.pos = pos
        self.neg = neg
        self.lam = lam

    def forward(self, output, target):
        # output, 128 x 10
        # print(output.size())
        gt = Variable(torch.zeros(output.size(0), self.num_classes), requires_grad=False)
        gt = gt.scatter_(1, target.unsqueeze(1), 1)
        zero = Variable(torch.zeros(1))
        pos_part = torch.max(zero, self.pos - output).pow(2)
        neg_part = torch.max(zero, output - self.neg).pow(2)
        loss = gt * pos_part + self.lam * (1-gt) * neg_part
        return loss.sum() / output.size(0)


class SpreadLoss(nn.Module):
    def __init__(self, opt, num_classes,
                 m_low=0.2, m_high=0.9, margin_epoch=20,
                 fix_m=False):
        super(SpreadLoss, self).__init__()
        self.num_classes = num_classes
        self.total_epoch = opt.max_epoch
        self.m_low = m_low
        self.m_high = m_high
        self.margin_epoch = margin_epoch
        self.interval = (m_high - m_low) / (self.total_epoch - 2*margin_epoch)
        self.fix_m = fix_m

    def forward(self, output, target, epoch):

        bs = output.size(0)
        if self.fix_m:
            m = self.m_high
        else:
            if epoch < self.margin_epoch:
                m = self.m_low
            elif epoch >= self.total_epoch - self.margin_epoch:
                m = self.m_high
            else:
                m = self.m_low + self.interval * (epoch - self.margin_epoch)

        target_output = torch.stack([output[i, target[i].data[0]] for i in range(bs)])
        loss = output - target_output + m
        gt = Variable(torch.zeros(output.size(0), self.num_classes), requires_grad=False)
        gt = gt.scatter_(1, target.unsqueeze(1), 1)
        loss = loss * (1-gt)
        loss = torch.clamp(loss, min=0).pow(2)   # 128 x 10
        return loss.sum() / output.size(0)
