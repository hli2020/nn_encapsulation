import argparse
import random
from utils.utils import *


class Options(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='Capsule Network')
        self.parser.add_argument('--experiment_name', default='base')
        self.parser.add_argument('--base_save_folder', default='result')
        self.parser.add_argument('--dataset', default='cifar10',
                                 help='[ cifar10/cifar100/mnist/svhn/fmnist | tiny_imagenet ]')
        self.parser.add_argument('--less_data_aug', action='store_true', help='see create_dset.py')
        # only valid for imagenet
        self.parser.add_argument('--setting', default='top1', type=str, help='[ top1 | top5 | obj_det ]')
        self.parser.add_argument('--bigger_input', action='store_true', help='only valid for imagenet')

        self.parser.add_argument('--debug_mode', default=True, type=str2bool,
                                 help='if true, single-gpu mode; display more loss/error per iteration')
        self.parser.add_argument('--measure_time', action='store_true')
        self.parser.add_argument('--s35', action='store_true', help='run on server 2035')
        self.parser.add_argument('--manual_seed', default=-1, type=int)
        self.parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--no_visdom', action='store_true')
        self.parser.add_argument('--port_id', default=8000, type=int)
        self.parser.add_argument('--device_id', default='0,1', type=str)

        # model params
        # if net_config == 'xx_default', all configs below matter;
        # otherwise, see details in 'net_config.py'
        self.parser.add_argument('--net_config', default='capnet_default', type=str,
                                 help='[xx_default | set1 |...| set_OT]')
        # RESNET
        self.parser.add_argument('--depth', default=14, type=int)  # 14 or 20, ...

        # CAPNET
        self.parser.add_argument('--route', default='dynamic', type=str)
        self.parser.add_argument('--E_step_norm', action='store_true')
        self.parser.add_argument('--route_num', default=3, type=int)
        self.parser.add_argument('--primary_cap_num', default=32, type=int)
        self.parser.add_argument('--b_init', default='zero', type=str, help='[zero | rand | learn]')
        self.parser.add_argument('--squash_manner', default='paper', type=str, help='[sigmoid|paper]')
        # if comp_cap=True, replace the capLayer with FC layer
        self.parser.add_argument('--comp_cap', action='store_true')

        # ENCAPNET
        self.parser.add_argument('--cap_N', default=4, type=int, help='multiple capLayers')
        self.parser.add_argument('--connect_detail', default='all', type=str,
                                 help='residual connections, [default | only_sub | all]')
        self.parser.add_argument('--fc_manner', default='default', type=str)
        self.parser.add_argument('--layerwise', action='store_true')
        self.parser.add_argument('--wider', action='store_true')
        # capRoute scheme, manner=0, 3, 4, ...
        self.parser.add_argument('--manner', default='3', type=str, help='str')
        self.parser.add_argument('--coeff_dimwise', action='store_true')  # TODO(low)
        self.parser.add_argument('--use_capBN', action='store_true')
        self.parser.add_argument('--skip_relu', action='store_true')

        # train
        self.parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
        # self.parser.add_argument('--scheduler', default=None, help='plateau, multi_step')
        self.parser.add_argument('--optim', default='adam', type=str)
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--weight_decay', default=5e-4, type=float)
        self.parser.add_argument('--gamma', default=0.1, type=float)  # for step lr scheme
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')

        self.parser.add_argument('--batch_size_train', default=32, type=int)
        # self.parser.add_argument('--batch_size_test', default=128, type=int)
        self.parser.add_argument('--max_epoch', default=600, type=int, help='Number of training epoches')
        self.parser.add_argument('--schedule', default=[200, 300, 400], nargs='+', type=int)
        self.parser.add_argument('--ot_loss_fac', default=1.0, type=float)
        # OT arguments
        # the following is defined on a case-by-case basis(net_config.py, like 'set_OT')
        self.parser.add_argument('--withCapRoute', action='store_true')
        self.parser.add_argument('--remove_bias', action='store_true')
        self.parser.add_argument('--encapsulate_G', action='store_true')
        self.parser.add_argument('--C_form', type=str, default='l2', help='l2 | cosine')
        self.parser.add_argument('--skip_critic', action='store_true')
        self.parser.add_argument('--no_bp_P_L', action='store_true')
        # cls loss
        self.parser.add_argument('--loss_fac', default=1.0, type=float)  # make loss larger
        self.parser.add_argument('--loss_form', default='margin', type=str, help='[ CE | spread | margin ]')
        self.parser.add_argument('--fix_m', action='store_true', help='valid for use_spread_loss only')
        # preserved for legacy reason (no more KL loss)
        self.parser.add_argument('--use_KL', action='store_true')
        self.parser.add_argument('--KL_manner', default=1, type=int)
        self.parser.add_argument('--KL_factor', default=.1, type=float)
        # stacking std, mean info of multiple capLayers (in CapLayer2 class), to compute KL loss
        self.parser.add_argument('--use_multiple', action='store_true', help='valid for N > 1')

        # test
        self.parser.add_argument('--show_test_after_epoch', type=int, default=-1)
        self.parser.add_argument('--multi_crop_test', action='store_true')
        # show stats
        self.parser.add_argument('--draw_hist', action='store_true')
        self.parser.add_argument('--non_target_j', action='store_true')

        self.opt = self.parser.parse_args()
        self.opt.phase = 'train_val'

    def setup_config(self):

        self.opt.batch_size_test = self.opt.batch_size_train
        self.opt.device_id = [int(x) for x in self.opt.device_id.split(',')]
        self.opt.save_folder = os.path.join(
            self.opt.base_save_folder, self.opt.experiment_name)
        if not os.path.exists(self.opt.save_folder):
            mkdirs(self.opt.save_folder)

        seed = random.randint(1, 10000) if self.opt.manual_seed == -1 else self.opt.manual_seed
        self.opt.random_seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        if self.opt.s35:
            self.opt.port_id = 9000

        if torch.cuda.is_available():
            self.opt.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # self.opt.device_id = torch.cuda.current_device()
        else:
            self.opt.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')

        if self.opt.debug_mode:
            self.opt.show_test_after_epoch = 0
            self.opt.show_freq = 1
            self.opt.save_epoch = 1
        else:
            if self.opt.show_test_after_epoch == -1:
                self.opt.show_test_after_epoch = 100
            self.opt.show_freq = 100
            self.opt.save_epoch = 25
        self._sort_up_attr()

    def _sort_up_attr(self):
        options = []
        if self.opt.dataset != 'tiny_imagenet':
            options.extend(['setting', 'bigger_input'])

        if self.opt.net_config[0:6] != 'resnet':
            options.extend(['depth'])

        if self.opt.net_config[0:6] != 'capnet':
            options.extend(['E_step_norm', 'route', 'route_num', 'primary_cap_num',
                            'b_init', 'squash_manner', 'comp_cap'])

        if self.opt.net_config[0:8] != 'encapnet':
            options.extend(['cap_N', 'connect_detail', 'fc_manner',
                            'layerwise', 'wider', 'manner',
                            'coeff_dimwise', 'use_capBN', 'skip_relu',
                            ])
        if self.opt.loss_form != 'spread':
            options.extend(['fix_m'])

        options.extend(['use_KL', 'KL_manner', 'KL_factor', 'use_multiple'])
        self._delete_attr(options)

    def _delete_attr(self, options):
        for option in options:
            delattr(self.opt, option)
