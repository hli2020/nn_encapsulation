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
args.experiment_name = ''
args.cap_model = 'v0'
args.test_model_list = [1, 20, 80, 200, 300]

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



# set loss
if args.model_cifar == 'capsule':
    if args.use_CE_loss:
        criterion = nn.CrossEntropyLoss()
    elif args.use_spread_loss:
        criterion = SpreadLoss(args, fix_m=args.fix_m,
                               num_classes=test_loader.dataset.num_classes)
    else:
        # default loss
        criterion = MarginLoss(num_classes=test_loader.dataset.num_classes)
elif args.model_cifar == 'resnet':
    criterion = nn.CrossEntropyLoss()
if args.use_cuda:
    criterion, model = criterion.cuda(), model.cuda()
cudnn.benchmark = True



for _, i in enumerate(test_model_list):
    model_file = \
        os.path.join(args.save_folder, 'epoch_{:d}.pth'.format(i))
    print('loading weights of model [{:s}]'.format(os.path.basename(model_file)))
    model = load_weights(model_file, model)
    args.cifar_model = model_file
    info = test(test_loader, model, criterion, args, vis)
    print('test acc is {:.4f}'.format(info['test_acc']))


# if args.test_only:
#     test_model_list = [1, 20, 80, 200, 300]
#
#     for _, i in enumerate(test_model_list):
#         model_file = \
#             os.path.join(args.save_folder, 'epoch_{:d}.pth'.format(i))
#         print('loading weights of model [{:s}]'.format(os.path.basename(model_file)))
#         model = load_weights(model_file, model)
#         args.cifar_model = model_file
#         info = test(test_loader, model, criterion, args, vis)
#         print('test acc is {:.4f}'.format(info['test_acc']))





