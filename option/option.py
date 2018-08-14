import argparse
import random
from utils.utils import *
import torch


class Options(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='Neural Network Encapsulation')
        self.parser.add_argument('--experiment_name', default='default')
        self.parser.add_argument('--base_save_folder', default='result')
        self.parser.add_argument('--dataset', default='cifar10',
                                 help='[ cifar10/cifar100/mnist/svhn/fmnist ]')
        self.parser.add_argument('--less_data_aug', action='store_true', help='see create_dset.py')

        self.parser.add_argument('--manual_seed', default=-1, type=int)
        self.parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data loading')
        self.parser.add_argument('--no_visdom', action='store_false')
        self.parser.add_argument('--port_id', default=8000, type=int)
        self.parser.add_argument('--device_id', default='0', type=str)

        # MODEL PARAMS
        # for a full list of options, see build_net (argument "type") in the layers/models/net_config.py file
        self.parser.add_argument('--net_config', default='encapnet_set_OT', type=str,
                                 help='[ encapnet_set_OT | resnet_default | ... | capnet_default ]')
        # RESNET
        self.parser.add_argument('--depth', default=14, type=int)  # 14 or 20, ...

        # CAPNET
        self.parser.add_argument('--route', default='dynamic', type=str)
        self.parser.add_argument('--E_step_norm', action='store_true')
        self.parser.add_argument('--route_num', default=3, type=int)
        self.parser.add_argument('--primary_cap_num', default=32, type=int)
        self.parser.add_argument('--b_init', default='zero', type=str, help='[zero | rand | learn]')
        self.parser.add_argument('--squash_manner', default='paper', type=str, help='[sigmoid|paper]')

        # TRAIN DYNAMICS
        self.parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
        self.parser.add_argument('--optim', default='adam', type=str)
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--weight_decay', default=5e-4, type=float)
        self.parser.add_argument('--gamma', default=0.1, type=float)  # for step lr scheme
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--batch_size_train', default=256, type=int)
        self.parser.add_argument('--max_epoch', default=600, type=int, help='Number of training epoches')
        self.parser.add_argument('--schedule', default=[200, 300, 400], nargs='+', type=int)

        # OT arguments
        self.parser.add_argument('--ot_loss_fac', default=1.0, type=float)
        self.parser.add_argument('--withCapRoute', action='store_true')
        self.parser.add_argument('--remove_bias', action='store_true')
        self.parser.add_argument('--encapsulate_G', action='store_true')
        self.parser.add_argument('--C_form', type=str, default='l2', help='l2 | cosine')
        self.parser.add_argument('--skip_critic', action='store_true')
        self.parser.add_argument('--no_bp_P_L', action='store_true')

        # cls loss
        self.parser.add_argument('--loss_fac', default=1.0, type=float)  # make loss larger
        self.parser.add_argument('--loss_form', default='margin', type=str, help='[ CE | spread | margin ]')
        self.parser.add_argument('--fix_m', action='store_true', help='valid for spread loss only')

        # TEST OPTIONS
        self.parser.add_argument('--show_test_after_epoch', type=int, default=0)
        self.parser.add_argument('--multi_crop_test', action='store_true')

        self.opt = self.parser.parse_args()
        self.opt.phase = 'train_val'

    def setup_config(self):

        if torch.__version__.startswith('0.4'):
            self.opt.pt_new = True
        elif torch.__version__.startswith('0.3'):
            self.opt.pt_new = False

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

        if torch.cuda.is_available():
            self.opt.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.opt.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')

        if self.opt.pt_new:
            self.opt.device = 'cpu' if not self.opt.use_cuda else 'cuda'

        if self.opt.show_test_after_epoch == -1:
            self.opt.show_test_after_epoch = 100
        self.opt.show_freq = 100
        self.opt.save_epoch = 25
        self._sort_up_attr()

    def _sort_up_attr(self):
        opts_to_del = []

        if self.opt.net_config[0:6] != 'resnet':
            opts_to_del.extend(['depth'])

        if self.opt.net_config[0:6] != 'capnet':
            opts_to_del.extend(['route', 'E_step_norm', 'route_num', 'primary_cap_num',
                                'b_init', 'squash_manner'])

        if not self.opt.net_config.startswith('encapnet'):
            opts_to_del.extend(['ot_loss_fac', 'withCapRoute', 'remove_bias', 'encapsulate_G',
                                'C_form', 'skip_critic', 'no_bp_P_L'])

        if self.opt.loss_form != 'spread':
            opts_to_del.extend(['fix_m'])

        self._delete_attr(opts_to_del)

    def _delete_attr(self, options):
        for option in options:
            delattr(self.opt, option)
