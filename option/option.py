import argparse
import random
from object_detection.utils.util import *


class Options(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='Capsule Network')
        self.parser.add_argument('--experiment_name', default='base')
        self.parser.add_argument('--base_save_folder', default='result')
        self.parser.add_argument('--dataset', default='cifar10', help='[ cifar10 | tiny_imagenet ]')
        # only valid for imagenet
        self.parser.add_argument('--setting', default='top1', type=str, help='[ top1 | top5 | obj_det ]')
        self.parser.add_argument('--bigger_input', action='store_true', help='only valid for imagenet')
        self.parser.add_argument('--less_data_aug', action='store_true', help='see create_dset.py')

        self.parser.add_argument('--debug_mode', default=True, type=str2bool)
        self.parser.add_argument('--measure_time', action='store_true')

        self.parser.add_argument('--s35', action='store_true', help='run on server 2035')
        self.parser.add_argument('--manual_seed', default=-1, type=int)
        self.parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--no_visdom', action='store_true')
        self.parser.add_argument('--port_id', default=8000, type=int)
        self.parser.add_argument('--device_id', default='0', type=str)

        # model params
        self.parser.add_argument('--cap_model', default='v1_2', type=str, help='v_base, v0, ...')

        # only valid for cap_model=v_base
        self.parser.add_argument('--depth', default=14, type=int)

        # only valid for cap_model=v0:
        self.parser.add_argument('--route_num', default=3, type=int)
        self.parser.add_argument('--primary_cap_num', default=32, type=int)
        self.parser.add_argument('--pre_ch_num', default=256, type=int)

        self.parser.add_argument('--add_cap_dropout', action='store_true')
        self.parser.add_argument('--dropout_p', default=0.2, type=float)
        self.parser.add_argument('--add_cap_BN_relu', action='store_true')
        self.parser.add_argument('--use_instanceBN', action='store_true')

        self.parser.add_argument('--b_init', default='zero', type=str, help='[zero | rand | learn]')
        self.parser.add_argument('--squash_manner', default='paper', type=str, help='[sigmoid|paper]')
        # if comp_cap=True, replace the capLayer with FC layer
        self.parser.add_argument('--comp_cap', action='store_true')

        # only valid for cap_model=v1_x
        self.parser.add_argument('--cap_N', default=3, type=int, help='multiple capLayers')
        # only valid for cap_model=v1_2_x
        self.parser.add_argument('--fc_manner', default='default', type=str)

        # train
        self.parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
        # TODO(low): add lr scheme
        # self.parser.add_argument('--scheduler', default=None, help='plateau, multi_step')
        self.parser.add_argument('--optim', default='rmsprop', type=str)
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--weight_decay', default=5e-4, type=float)
        self.parser.add_argument('--gamma', default=0.1, type=float)  # for step lr scheme
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')

        self.parser.add_argument('--batch_size_train', default=100, type=int)
        self.parser.add_argument('--batch_size_test', default=100, type=int)
        self.parser.add_argument('--max_epoch', default=300, type=int, help='Number of training epoches')
        self.parser.add_argument('--schedule', default=[150, 225], nargs='+', type=int)
        # loss
        self.parser.add_argument('--loss_form', default='CE', type=str, help='[ CE | spread | margin ]')
        # preserved for legacy reason (no more KL loss)
        self.parser.add_argument('--use_KL', action='store_true')
        self.parser.add_argument('--KL_manner', default=1, type=int)
        self.parser.add_argument('--KL_factor', default=.1, type=float)
        # stacking std, mean info of multiple capLayers (in CapLayer2 class), to compute KL loss
        self.parser.add_argument('--use_multiple', action='store_true', help='valid for N > 1')
        self.parser.add_argument('--fix_m', action='store_true', help='valid for use_spread_loss only')

        # test
        self.parser.add_argument('--show_test_after_epoch', type=int, default=-1)
        self.parser.add_argument('--multi_crop_test', action='store_true')
        # show stats
        self.parser.add_argument('--draw_hist', action='store_true')
        self.parser.add_argument('--non_target_j', action='store_true')

        self.opt = self.parser.parse_args()
        self.opt.phase = 'train_val'

    def setup_config(self):

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
