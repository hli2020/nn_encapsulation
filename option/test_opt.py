import argparse
from utils.util import *


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--dataset', default='coco', help='[ voc | coco ]')
parser.add_argument('--experiment_name', default='ssd_base_101')
# parser.add_argument('--trained_model', default='final_v2.pth', type=str)
parser.add_argument('--trained_model', default='ssd512_COCO_iter_155.pth', type=str)
parser.add_argument('--sub_folder_suffix', default='', type=str)

# model params
parser.add_argument('--ssd_dim', default=512, type=int)
parser.add_argument('--prior_config', default='v2_512', type=str)

# for test
parser.add_argument('--soft_nms', type=int, default=-1)  # set -1 if not used
parser.add_argument('--conf_thresh', default=0.05, type=float, help='Detection confidence threshold')
parser.add_argument('--top_k', default=300, type=int, help='The Maximum number of box preds to consider in NMS.')
parser.add_argument('--nms_thresh', default=0.5, type=float)
parser.add_argument('--show_freq', default=10, type=int, help='show freq in console')

# runtime and display
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--port_id', default=8097, type=int)
parser.add_argument('--display_id', default=1, type=int)
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')

args = parser.parse_args()
args.phase = 'test'

_temp = '' if args.sub_folder_suffix == '' else '_'
args.save_folder = os.path.join('result', args.experiment_name, args.phase,
                                (args.trained_model + _temp + args.sub_folder_suffix))
args.trained_model = os.path.join('result', args.experiment_name, 'train', args.trained_model)

if not os.path.exists(args.save_folder):
    mkdirs(args.save_folder)

if torch.cuda.is_available():
    args.cuda = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

