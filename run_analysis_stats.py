from __future__ import print_function
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from layers.capsule import CapsNet
from layers.train_val import test

from data.create_dset import create_dataset
from object_detection.utils.visualizer import Visualizer
from object_detection.utils.util import *
from option.option import Options

cudnn.benchmark = True
# config
option = Options()
option.setup_config()
args = option.opt

# ==============================
# uncomment the following if you run in .sh file
args.experiment_name = 'base_101_v4_rerun'
args.cap_model = 'v0'
args.test_model_list = [1, 50, 100, 200, 300, 247]

args.draw_hist = True
args.test_only = True
args.non_target_j = False

args.look_into_details = False
# ==============================

# init
show_jot_opt(args)
visual = Visualizer(args)

test_loader = data.DataLoader(create_dataset(args, 'test'), args.batch_size_test,
                              num_workers=args.num_workers, shuffle=False, pin_memory=True)
# model
model = CapsNet(num_classes=test_loader.dataset.num_classes, opts=args)
model = model.cuda()

model_summary, param_num = torch_summarize(model)
print_log(model_summary, args.file_name)
print_log('Total param num # {:f} Mb'.format(param_num), args.file_name)

for i in args.test_model_list:
    model_file = \
        os.path.join(args.save_folder, 'epoch_{:d}.pth'.format(i))
    print('loading weights of model [{:s}]'.format(os.path.basename(model_file)))
    model = (model_file, model)
    args.cifar_model = model_file
    info = test(test_loader, model, criterion, args, vis)
    print('test acc is {:.4f}'.format(info['test_acc']))




