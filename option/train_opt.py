import argparse
import random
from utils.util import *

parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--experiment_name', default='ssd_rerun')
parser.add_argument('--dataset', default='coco',
                    help='[ voc | coco ]')
parser.add_argument('--deploy', action='store_true')
# args_temp = parser.parse_args()

parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--scheduler', default=None, help='plateau, multi_step')
parser.add_argument('--optim', default='rmsprop', type=str)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')

# VOC and COCO
# if args_temp.dataset == 'voc' or args_temp.dataset == 'coco':
# training config
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--max_iter', default=130000, type=int, help='Number of training iterations')
parser.add_argument('--no_pretrain', action='store_true', help='default is using pretrain')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
# parser.add_argument('--resume', default='ssd300_0712_iter_30', type=str, help='Resume from checkpoint')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
# TODO: MUST comment the schedule if you run CIFAR
parser.add_argument('--schedule', default=[80000, 100000, 120000], nargs='+')
# model params
parser.add_argument('--ssd_dim', default=512, type=int)
parser.add_argument('--prior_config', default='v2_512', type=str)

# CIFAR
# elif args_temp.dataset == 'cifar':
# for cifar only
# parser.add_argument('--model_cifar', default='capsule', type=str, help='resnet | capsule')
# parser.add_argument('--multi_crop_test', action='store_true')
# # network, v0 is the structure in the paper
# parser.add_argument('--cap_model', default='v4_1', type=str,
#                     help='only valid when model_cifar is [capsule], v0, v1, v2, v3')
# parser.add_argument('--cap_N', default=3, type=int)
# # loss
# parser.add_argument('--use_KL', action='store_true')
# parser.add_argument('--use_multiple', action='store_true', help='valid for N > 1')
# parser.add_argument('--KL_manner', default=1, type=int)
# parser.add_argument('--KL_factor', default=.1, type=float)
# parser.add_argument('--use_CE_loss', action='store_true')
# parser.add_argument('--use_spread_loss', action='store_true')
# parser.add_argument('--fix_m', action='store_true', help='valid for use_spread_loss only')
# # FOR cap_model=v0 only:
# parser.add_argument('--look_into_details', action='store_true')
# parser.add_argument('--add_cap_dropout', action='store_true')
# parser.add_argument('--dropout_p', default=0.2, type=float)
# parser.add_argument('--has_relu_in_W', action='store_true')
# parser.add_argument('--do_squash', action='store_true', help='for w_v3 alone')  # squash is much better
# parser.add_argument('--w_version', default='v2', type=str, help='[v0, v1, v2, v3]')
# parser.add_argument('--b_init', default='zero', type=str, help='[zero | rand | learn]')
# parser.add_argument('--squash_manner', default='sigmoid', type=str)
# # general for all cap_model
# parser.add_argument('--route_num', default=4, type=int)
# parser.add_argument('--epochs', default=300, type=int)
# parser.add_argument('--schedule_cifar', type=int, nargs='+', default=[150, 225],
#                     help='Decrease learning rate at these epochs.')
# parser.add_argument('--train_batch', default=128, type=int, metavar='N')
# parser.add_argument('--test_batch', default=128, type=int, metavar='N')
# parser.add_argument('--save_epoch', default=20, type=int)
# # keep the following for legacy purpose
# parser.add_argument('--draw_hist', action='store_true')
# parser.add_argument('--test_only', action='store_true')
# parser.add_argument('--non_target_j', action='store_true')

# RUNTIME AND DISPLAY
parser.add_argument('--manual_seed', default=-1, type=int)
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--port', default=4000, type=int)
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
args = parser.parse_args()

# GENERAL SETTING
# TODO: change ssd iter to epoch training
args.start_epoch = 1
args.max_epoch = args.epochs if hasattr(args, 'epochs') else args.max_iter

args.debug = not args.deploy
# if args.deploy:
#     # when in deploy mode, we will use port_id = 2000 as default on server
#     args.port = 2000
args.phase = 'train'
args.save_folder = os.path.join('result', args.experiment_name, args.phase)

if args.dataset == 'voc' or args.dataset == 'coco':
    if args.resume:
        args.resume = os.path.join(args.save_folder, (args.resume + '.pth'))

    if type(args.schedule[0]) == str:
        temp_ = args.schedule[0].split(',')
        schedule = list()
        for i in range(len(temp_)):
            schedule.append(int(temp_[i]))
        args.schedule = schedule

    if args.debug:
        args.loss_freq, args.save_freq = 10, 20
    else:
        args.loss_freq, args.save_freq = 50, 5000

if args.dataset == 'cifar' and args.test_only:
    # for cifar only
    args.phase = 'test'
    args.max_epoch = 0

if not os.path.exists(args.save_folder):
    mkdirs(args.save_folder)

if torch.cuda.is_available():
    args.use_cuda = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    args.use_cuda = False
    torch.set_default_tensor_type('torch.FloatTensor')

if args.manual_seed == -1:
    args.manual_seed = random.randint(1, 10000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
