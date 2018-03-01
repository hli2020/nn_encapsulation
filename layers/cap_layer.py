import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import time
import numpy as np
import math
from layers.misc import compute_stats

EPS = np.finfo(float).eps
# to find the different prediction during routing
FIND_DIFF = False
# to see the detailed values of b, c, v, delta_b
LOOK_INTO_DETAILS = False


class CapLayer(nn.Module):
    """
        The original capsule implementation in the NIPS paper;
        and detailed ablative analysis on it.
    """
    def __init__(self, opts, num_in_caps, num_out_caps,
                 out_dim, in_dim, num_shared, route):
        super(CapLayer, self).__init__()

        self.route = route
        self.measure_time = opts.measure_time
        self.which_sample, self.which_j = 0, 0
        self.non_target_j = opts.non_target_j

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num_shared = num_shared
        self.route_num = opts.route_num
        self.comp_cap = opts.comp_cap
        self.num_out_caps = num_out_caps
        self.num_in_caps = num_in_caps

        self.use_KL = False     # opts.use_KL
        self.KL_manner = False  # opts.KL_manner
        self.add_cap_BN_relu = opts.add_cap_BN_relu
        self.add_cap_dropout = opts.add_cap_dropout
        self.squash_manner = opts.squash_manner

        if self.comp_cap:
            self.fc1 = nn.Linear(num_in_caps, num_in_caps)
            self.relu1 = nn.ReLU(inplace=True)
            self.drop1 = nn.Dropout(p=.5)
            self.fc2 = nn.Linear(num_in_caps, num_out_caps)
            self.fc_time = opts.fc_time
        else:
            self.W = nn.Conv2d(num_shared*in_dim, num_shared*num_out_caps*out_dim,
                               kernel_size=1, stride=1, groups=num_shared)
        if self.add_cap_dropout:
            self.cap_droput = nn.Dropout2d(p=opts.dropout_p)
        if self.add_cap_BN_relu:
            # self.cap_BN = nn.BatchNorm2d(out_dim)
            self.cap_BN = nn.InstanceNorm2d(out_dim, affine=True)
            self.cap_relu = nn.ReLU(True)
        if self.route == 'EM':
            self.beta_v = Parameter(torch.Tensor(self.num_out_caps))
            self.beta_a = Parameter(torch.Tensor(self.num_out_caps))
            nn.init.uniform(self.beta_v.data)
            nn.init.uniform(self.beta_a.data)
            # self.beta_v = 10
            # self.beta_a = -self.beta_v
            self.fac_R_init = 1.
            self.fac_temperature = 1.
            self.fac_nor = 10.0
            self.E_step_norm = opts.E_step_norm

    def forward(self, input,
                target=None, curr_iter=None, draw_hist=None,
                activate=None):
        """
            target, curr_iter, draw_hist are for debugging or stats collection purpose only;
            'activate' is the activation of previous layer, a_i (i being # of input capsules)
            'epoch' is for inverse temperature to compute lambda
            return:
                v, stats, activation (a_j)
        """
        start, activation = [], []
        # for draw_hist (test)
        batch_cos_dist, batch_i_length, batch_cos_v, avg_len = [], [], [], []
        # for KL loss (train)
        mean, std = [], []

        bs, in_channels, h, w = input.size()
        pred_list = []
        if FIND_DIFF:
            pred_list.extend(list(range(bs)))
            pred_list.extend(target.data)

        if self.comp_cap:
            x = input.view(input.size(0), -1)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.drop1(x)
            v = self.fc2(x)
        else:
            # 1. affine mapping (get 'pred')
            if self.measure_time:
                torch.cuda.synchronize()
                start = time.perf_counter()

            pred = self.W(input)
            # pred: bs, 5120, 6, 6
            # -> bs, 10, 16, 32x6x6 (view)
            spatial_size = pred.size(2)
            pred = pred.view(bs, self.num_shared, self.num_out_caps, self.out_dim,
                             spatial_size, spatial_size)
            pred = pred.permute(0, 2, 3, 1, 4, 5).contiguous()
            pred = pred.view(bs, pred.size(1), pred.size(2), -1)
            # if self.add_cap_BN_relu:
            #     NotImplementedError()
            #     # pred = self.cap_BN(pred.permute(0, 3, 1, 2).contiguous())
            #     # pred = pred.permute(0, 2, 3, 1)
            #     # pred = self.cap_relu(pred)
            # if self.add_cap_dropout:
            #     NotImplementedError()
            #     # v = self.cap_droput(v)
            if self.measure_time:
                torch.cuda.synchronize()
                print('\tcap W time: {:.4f}'.format(time.perf_counter() - start))

            # 2. routing (get 'v')
            # _check = np.isnan(pred.data.cpu().numpy())
            # if _check.any():
            #     a = 1
            # print('\nlearned W mean: {:.4f}'.format(torch.mean(self.W.weight.data)))
            # print('learned b mean: {:.4f}'.format(torch.mean(self.W.bias.data)))
            # try:
            #     a = torch.mean(self.W.weight.grad.data)
            #     print('last iter saved grad:')
            #     print('learned W_grad mean: {:.6f}'.format(torch.mean(self.W.weight.grad.data)))
            #     print('learned b_grad mean: {:.6f}'.format(torch.mean(self.W.bias.grad.data)))
            # except:
            #     pass
            start = time.perf_counter()
            if self.route == 'dynamic':
                v = self.dynamic(bs, pred, pred_list)
            elif self.route == 'EM':
                v, activation = self.EM(bs, pred, activate)

            if self.measure_time:
                torch.cuda.synchronize()
                print('\tcap Route (r={:d}) time: {:.4f}'.format(self.route_num, time.perf_counter() - start))
            if draw_hist:
                batch_cos_dist, batch_i_length, batch_cos_v, avg_len = \
                    compute_stats(target, pred, v, self.non_target_j)
            if self.use_KL:
                mean, std = compute_stats(None, pred, v, KL_manner=self.KL_manner)
            if FIND_DIFF:
                temp = np.asarray(pred_list)
                temp = np.resize(temp, (self.route_num+2, bs)).transpose()  # 128 x 5
                predict_ = temp[:, 2:]
                check_ = np.sum((predict_ - predict_[:, 0].reshape(bs, 1)), axis=1)
                diff_ind = np.nonzero(check_)[0]
                print('curr_iter {:d}:'.format(curr_iter))
                if diff_ind.shape == (0,):
                    HAS_DIFF = False
                    print('no difference prediction during routing!')
                else:
                    HAS_DIFF = True
                    print(temp[diff_ind, :])
                    print('\n')
        stats = [
            batch_cos_dist, batch_i_length, batch_cos_v, avg_len,
            mean, std
        ]
        return v, stats, activation

    def dynamic(self, bs, pred, pred_list):
        # create b on the fly
        # if opts.b_init == 'rand':
        #     self.b = Variable(torch.rand(num_out_caps, num_in_caps), requires_grad=False)
        # elif opts.b_init == 'zero':
        #     self.b = Variable(torch.zeros(num_out_caps, num_in_caps), requires_grad=False)
        # elif opts.b_init == 'learn':
        #     self.b = Variable(torch.zeros(num_out_caps, num_in_caps), requires_grad=True)
        b = Variable(torch.zeros(bs, self.num_out_caps, self.num_in_caps),
                     requires_grad=False)

        for i in range(self.route_num):

            internal_start = time.perf_counter()
            c = F.softmax(b, dim=1)
            if self.measure_time:
                torch.cuda.synchronize()
                b_sftmax_t = time.perf_counter() - internal_start
                t1 = time.perf_counter()
            # TODO: optimize time (0.0238)
            s = torch.matmul(pred, c.unsqueeze(3)).squeeze()
            if self.measure_time:
                torch.cuda.synchronize()
                s_matmul_t = time.perf_counter() - t1
                t2 = time.perf_counter()

            # shape of v: bs, out_cap_num, out_cap_dim
            v = squash(s, self.squash_manner)
            if self.measure_time:
                torch.cuda.synchronize()
                squash_t = time.perf_counter() - t2
                t3 = time.perf_counter()

            # update b/c
            if i < self.route_num - 1:
                if self.measure_time:
                    torch.cuda.synchronize()
                    permute_t = time.perf_counter() - t3
                    t4 = time.perf_counter()
                # TODO: super inefficient (0.1557s)
                # delta_b = torch.matmul(_pred, v.unsqueeze(3)).squeeze()
                delta_b = torch.matmul(v.unsqueeze(2), pred).squeeze()
                if self.measure_time:
                    torch.cuda.synchronize()
                    delta_matmul_t = time.perf_counter() - t4
                    t5 = time.perf_counter()

                if FIND_DIFF:
                    v_all_classes = v.norm(dim=2)
                    _, curr_pred = torch.max(v_all_classes, 1)
                    pred_list.extend(curr_pred.data)

                b = torch.add(b, delta_b)
                if self.measure_time:
                    torch.cuda.synchronize()
                    add_t = time.perf_counter() - t5

                if self.measure_time:
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    update_t = time.perf_counter() - t3
            else:
                update_t, add_t, delta_matmul_t, permute_t = 0, 0, 0, 0

            if self.measure_time:
                torch.cuda.synchronize()
                print('\t\tRoute r={:d} time: {:.4f}'.format(i, time.perf_counter() - internal_start))
                print('\t\t\tb_sftmax_t time: {:.4f}'.format(b_sftmax_t))
                print('\t\t\ts_matmul_t time: {:.4f}'.format(s_matmul_t))
                print('\t\t\tsquash_t time: {:.4f}'.format(squash_t))
                print('\t\t\tupdate_t time: {:.4f}'.format(update_t))
                print('\t\t\t\tpermute: {:.4f}'.format(permute_t))
                print('\t\t\t\tdelta_matmul: {:.4f}'.format(delta_matmul_t))
                print('\t\t\t\tadd: {:.4f}'.format(add_t))
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

            # print('activation, iter={:d}, min: {:.3f}, max: {:.3f}'.format(
            #     i, activation.data.min(), activation.data.max()
            # ))
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

        # _check = np.isnan(activation.data.cpu().numpy())
        # if _check.any():
        #     a = 1
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
            |   | BN-ReLU-Squash
            |   | conv2d
            |   | BN-ReLU-Squash
            |   | ... (N times)
            |   | conv2d
            | + | (skip connection)
            |   | BN-ReLU-Squash
        """
    def __init__(self, ch_num, groups,
                 N=1, ch_out=-1,
                 kernel_size=(1,), stride=1, pad=(0,), residual=False,
                 manner='0'):
        super(CapConv, self).__init__()
        self.ch_num_in = ch_num
        self.ch_num_out = ch_num if ch_out == -1 else ch_out
        self.groups = groups
        self.iter_N = N
        self.residual = residual
        self.wider_conv = True if len(kernel_size) >= 2 else False
        self.manner = manner

        if self.residual and self.ch_num_in != self.ch_num_out:
            self.conv_adjust_blob_shape = \
                nn.Conv2d(self.ch_num_in, self.ch_num_out,
                          kernel_size=3, padding=1, stride=stride)

        layers = []
        for i in range(self.iter_N):
            layers.append(_make_core_conv(
                manner=manner, wider_conv=self.wider_conv,
                ch_num_in=self.ch_num_in, ch_num_out=self.ch_num_out,
                kernel_size=kernel_size, stride=stride, groups=self.groups, pad=pad))
            # TODO: change BN per capsule, along the channel
            # TODO: investigate whether need BN and relu
            if i < self.iter_N-1:
                if manner == '0':
                    layers.append(nn.BatchNorm2d(self.ch_num_out))
                    layers.append(nn.ReLU(True))
                layers.append(conv_squash(self.groups))

        self.block = nn.Sequential(*layers)
        if manner == '0':
            self.last_bn = nn.BatchNorm2d(self.ch_num_out)
            self.last_relu = nn.ReLU()
        self.last_squash = conv_squash(self.groups)

    def forward(self, input):

        out = self.block(input)
        if self.residual:
            if hasattr(self, 'conv_adjust_blob_shape'):
                input = self.conv_adjust_blob_shape(input)
            out += input
        if self.manner == '0':
            out = self.last_bn(out)
            out = self.last_relu(out)
        out = self.last_squash(out)
        return out


class CapConv2(nn.Module):
    """
        Wrap up the CapConv layer with multiple sub layers into a module,
        possibly with skip connection.
    """
    def __init__(self, ch_in, ch_out, groups,
                 residual, iter_N,
                 no_downsample=False,               # for main_conv
                 layerwise_skip_connect=True,       # for sub_conv
                 more_skip=False,
                 wider_main_conv=False,             # for main_conv
                 manner='0',                        # for both
                 ):
        super(CapConv2, self).__init__()
        assert len(residual) == 2
        if more_skip:
            assert iter_N >= 2
            iter_N -= 1
        self.more_skip = more_skip

        # define main_conv
        # wider switch
        main_ksize = (5, 3, 1) if wider_main_conv else (3,)
        main_pad = (2, 1, 0) if wider_main_conv else (1,)
        main_stride = 1 if no_downsample else 2
        self.main_conv = CapConv(ch_num=ch_in, ch_out=ch_out, groups=groups,
                                 kernel_size=main_ksize, stride=main_stride,
                                 pad=main_pad, residual=residual[0], manner=manner)
        # define sub_conv (stride/pad/ksize are set by default)
        # layerwise switch
        if layerwise_skip_connect:
            layers = []
            for i in range(iter_N):
                layers.append(CapConv(ch_num=ch_out, groups=groups,
                                      residual=residual[1], manner=manner))
            self.sub_conv = nn.Sequential(*layers)
        else:
            "should be exactly the same as 'v1_3' in network.py"
            self.sub_conv = CapConv(ch_num=ch_out, groups=groups, N=iter_N,
                                    residual=residual[1], manner=manner)

        # define more_skip (optional)
        # more_skip switch
        if more_skip:
            # 'ms' means 'more_skip'
            if ch_in != ch_out:
                self.ms_conv_adjust_blob_shape = \
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=1)
            self.ms_conv = \
                _make_core_conv(manner=manner, ch_num_in=ch_out, ch_num_out=ch_out,
                                kernel_size=(1,), groups=groups, stride=1, pad=(0,))
            self.ms_bn = nn.BatchNorm2d(ch_out)
            self.ms_relu = nn.ReLU()
            self.ms_squash = conv_squash(groups)

    def forward(self, input):
        out = self.main_conv(input)
        out = self.sub_conv(out)
        if self.more_skip:
            out = self.ms_conv(out)
            if hasattr(self, 'ms_conv_adjust_blob_shape'):
                input = self.ms_conv_adjust_blob_shape(input)
            out += input
            out = self.ms_bn(out)
            out = self.ms_relu(out)
            out = self.ms_squash(out)
        return out


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


# Utilities below
def normalize_cap(input, factor):
    "input: bs, j, (i)"
    min_input, _ = torch.min(input, dim=1, keepdim=True)
    max_input, _ = torch.max(input, dim=1, keepdim=True)
    output = \
        factor*(input - min_input) / (max_input - min_input + EPS) - factor/2

    return output


def _make_core_conv(
        ch_num_in, ch_num_out, kernel_size, stride, groups, pad,
        manner='0', wider_conv=False):
    """
        used in convCap/convCap2 block
        kernel_size, pad, should be tuple type
    """
    conv_opt =[]
    if manner == '0':
        if wider_conv:
            # TODO (easy): merge wider case with capRoute below
            conv_opt = multi_conv(ch_num_in, ch_num_out,
                                  ksize=kernel_size, stride=stride,
                                  group=groups, pad=pad)
        else:
            conv_opt = nn.Conv2d(ch_num_in, ch_num_out,
                                 kernel_size=kernel_size[0], stride=stride,
                                 groups=groups, padding=pad[0])
    elif manner == '1':
        conv_opt = capConvRoute1(ch_num_in, ch_num_out,
                                 ksize=kernel_size, stride=stride,
                                 group=groups, pad=pad)
    elif manner == '2':
        conv_opt = capConvRoute2(ch_num_in, ch_num_out,
                                 ksize=kernel_size, stride=stride,
                                 group=groups, pad=pad)
    return conv_opt


def squash(vec, manner='paper'):
    """given a 3-D (bs, num_cap, dim_cap) input, squash the capsules
    output has the same shape as input
    """
    EPS, coeff2, mean = 1e-20, [], []

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


class capConvRoute1(nn.Module):
    """
        initial version of convCap with routing;
        used in '_make_core_conv' method to build basic convCap
    """
    def __init__(self,
                 ch_num_in, ch_num_out,
                 ksize, pad, stride, group):
        super(capConvRoute1, self).__init__()
        self.expand_factor = int(ch_num_out/group)
        self.group = group

        self.main_cap = nn.Sequential(*[
            nn.Conv2d(ch_num_in, ch_num_out, kernel_size=ksize[0],
                      stride=stride, groups=group, padding=pad[0]),
            nn.BatchNorm2d(ch_num_out),
            nn.ReLU(),
            conv_squash(group)
        ])
        # take the output of main_cap as input
        self.main_cap_coeff = nn.Conv2d(
            ch_num_out, group, kernel_size=3,
            stride=1, padding=1, groups=group)
        # res_cap: take the input as input; NO GROUPING
        self.res_cap = nn.Sequential(*[
            nn.Conv2d(ch_num_in, ch_num_out, kernel_size=ksize[0],
                      stride=stride, padding=pad[0]),
            nn.BatchNorm2d(ch_num_out),
            nn.ReLU(),
            conv_squash(group)
        ])

    def forward(self, x):
        main_out = self.main_cap(x)
        main_coeff = self.main_cap_coeff(main_out)
        main_coeff = torch.cat(
            [main_coeff[:, i, :, :].unsqueeze(dim=1).repeat(1, self.expand_factor, 1, 1)
             for i in range(self.group)], dim=1)

        res_out = self.res_cap(x)
        res_coeff = 1 - main_coeff
        out = main_out * main_coeff + res_out * res_coeff
        return out


class capConvRoute2(capConvRoute1):
    def __init__(self,
                 ch_num_in, ch_num_out,
                 ksize, pad, stride, group):
        super(capConvRoute2, self).__init__(
            ch_num_in, ch_num_out, ksize, pad, stride, group)
        self.main_cap_coeff = nn.Conv2d(
            ch_num_in, group, kernel_size=ksize[0],
            stride=stride, groups=group, padding=pad[0])

    def forward(self, x):
        main_out = self.main_cap(x)
        main_coeff = self.main_cap_coeff(x)
        main_coeff = torch.cat(
            [main_coeff[:, i, :, :].unsqueeze(dim=1).repeat(1, self.expand_factor, 1, 1)
             for i in range(self.group)], dim=1)

        res_out = self.res_cap(x)
        res_coeff = 1 - main_coeff
        out = main_out * main_coeff + res_out * res_coeff
        return out


class multi_conv(nn.Module):
    """
        used in '_make_core_conv' method to build parallel convolutions
    """
    def __init__(self,
                 ch_num_in, ch_num_out,
                 ksize, pad, stride, group):
        super(multi_conv, self).__init__()
        assert len(ksize) == len(pad)
        self.ch_num_in = ch_num_in
        self.ch_num_out = ch_num_out
        self.ksize = ksize
        self.pad = pad
        self.stride = stride
        self.group = group
        self.multi_N = len(ksize)

        self.multi1 = nn.Conv2d(ch_num_in, ch_num_out,
                                kernel_size=ksize[0], stride=stride,
                                groups=group, padding=pad[0])
        # http://pytorch.org/docs/master/nn.html#torch.nn.ModuleList
        if self.multi_N >= 2:
            self.multi2 = nn.Conv2d(ch_num_in, ch_num_out,
                                    kernel_size=ksize[1], stride=stride,
                                    groups=group, padding=pad[1])
            self.multi3 = nn.Conv2d(ch_num_in, ch_num_out,
                                    kernel_size=ksize[2], stride=stride,
                                    groups=group, padding=pad[2])

    def forward(self, input):

        # for i in range(self.multi_N):
        #     out = self.layers[i](input)
        #     if i == 0:
        #         out_sum = out
        #     else:
        #         out_sum += out
        out_sum = self.multi1(input)
        if self.multi_N >= 2:
            out_sum += self.multi2(input)
            out_sum += self.multi3(input)
        return out_sum

    # def __repr__(self):
    #     return self.__class__.__name__ + '(' \
    #         + 'ch_num_in=' + str(self.ch_num_in) \
    #         + ', ch_num_out=' + str(self.ch_num_out) \
    #         + ', ksize=' + str(self.ksize) \
    #         + ', stride=' + str(self.stride) \
    #         + ', group=' + str(self.group) \
    #         + ', pad=' + str(self.pad) + ')'


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
