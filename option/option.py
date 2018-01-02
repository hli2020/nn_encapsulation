import argparse
import random
# from utils.util import *
from object_detection.utils.util import *


class Options(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='Object Detection')
        self.parser.add_argument('--experiment_name', default='ssd_rerun')
        self.parser.add_argument('--dataset', default='coco', help='[ voc|coco ]')
        self.parser.add_argument('--debug_mode', default=True, type=str2bool)
        self.parser.add_argument('--base_save_folder', default='result')

        self.parser.add_argument('--manual_seed', default=-1, type=int)
        self.parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--no_visdom', action='store_true')
        self.parser.add_argument('--port_id', default=8090, type=int)

        # model params
        # network, v0 is the structure in the paper
        self.parser.add_argument('--cap_model', default='v4_1', type=str,
                                 help='only valid when model_cifar is [capsule], v0, v1, v2, v3')
        self.parser.add_argument('--cap_N', default=3, type=int)
        self.parser.add_argument('--route_num', default=4, type=int)
        # FOR cap_model=v0 only:
        self.parser.add_argument('--look_into_details', action='store_true')
        self.parser.add_argument('--add_cap_dropout', action='store_true')
        self.parser.add_argument('--dropout_p', default=0.2, type=float)
        self.parser.add_argument('--has_relu_in_W', action='store_true')
        self.parser.add_argument('--do_squash', action='store_true', help='for w_v3 alone')  # squash is much better
        self.parser.add_argument('--w_version', default='v2', type=str, help='[v0, v1, v2, v3]')
        self.parser.add_argument('--b_init', default='zero', type=str, help='[zero | rand | learn]')
        self.parser.add_argument('--squash_manner', default='sigmoid', type=str)

        # train
        self.parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
        self.parser.add_argument('--scheduler', default=None, help='plateau, multi_step')
        self.parser.add_argument('--optim', default='rmsprop', type=str)
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--weight_decay', default=5e-4, type=float)
        self.parser.add_argument('--gamma', default=0.1, type=float)
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')

        self.parser.add_argument('--batch_size_train', default=128, type=int)
        self.parser.add_argument('--batch_size_test', default=128, type=int)
        # self.parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
        self.parser.add_argument('--max_epoch', default=20, type=int, help='Number of training epoches')
        self.parser.add_argument('--schedule', default=[6, 12, 16], nargs='+', type=int)

        # loss
        self.parser.add_argument('--use_KL', action='store_true')
        self.parser.add_argument('--use_multiple', action='store_true', help='valid for N > 1')
        self.parser.add_argument('--KL_manner', default=1, type=int)
        self.parser.add_argument('--KL_factor', default=.1, type=float)
        self.parser.add_argument('--use_CE_loss', action='store_true')
        self.parser.add_argument('--use_spread_loss', action='store_true')
        self.parser.add_argument('--fix_m', action='store_true', help='valid for use_spread_loss only')

        # test
        self.parser.add_argument('--multi_crop_test', action='store_true')

        self.opt = self.parser.parse_args()

    def setup_config(self):

        self.opt.save_folder = os.path.join(self.opt.base_save_folder,
                                            self.opt.experiment_name, self.opt.phase)
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

        # TODO: test_after_epoch
        if self.opt.debug_mode:
            self.opt.loss_freq = 10
            self.opt.save_freq = self.opt.loss_freq
        else:
            self.opt.loss_freq = 100    # in iter unit
            self.opt.save_freq = 5      # in epoch unit


# # GENERAL SETTING
# args.start_epoch = 1
# args.max_epoch = args.epochs if hasattr(args, 'epochs') else args.max_iter
#
# args.debug = not args.deploy
# # if args.deploy:
# #     # when in deploy mode, we will use port_id = 2000 as default on server
# #     args.port = 2000
# args.phase = 'train'
# args.save_folder = os.path.join('result', args.experiment_name, args.phase)
#
# if args.dataset == 'voc' or args.dataset == 'coco':
#     if args.resume:
#         args.resume = os.path.join(args.save_folder, (args.resume + '.pth'))
#
#     if type(args.schedule[0]) == str:
#         temp_ = args.schedule[0].split(',')
#         schedule = list()
#         for i in range(len(temp_)):
#             schedule.append(int(temp_[i]))
#         args.schedule = schedule
#
#     if args.debug:
#         args.loss_freq, args.save_freq = 10, 20
#     else:
#         args.loss_freq, args.save_freq = 50, 5000
#
# if args.dataset == 'cifar' and args.test_only:
#     # for cifar only
#     args.phase = 'test'
#     args.max_epoch = 0
#
# if not os.path.exists(args.save_folder):
#     mkdirs(args.save_folder)
#
# if torch.cuda.is_available():
#     args.use_cuda = True
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     args.use_cuda = False
#     torch.set_default_tensor_type('torch.FloatTensor')
#
# if args.manual_seed == -1:
#     args.manual_seed = random.randint(1, 10000)
# random.seed(args.manual_seed)
# torch.manual_seed(args.manual_seed)
