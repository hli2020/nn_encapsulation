from __future__ import print_function
import torch.optim as optim
import torch.utils.data as data
from data.create_dset import create_dataset
from layers.capsule import CapsNet
from layers.cap_layer import MarginLoss, SpreadLoss
from layers.train_val import *

from object_detection.utils.visualizer import Visualizer
from object_detection.utils.util import *
from object_detection.utils.train import set_lr_schedule, adjust_learning_rate, save_checkpoint
from option.option import Options

# config
option = Options()
option.setup_config()
args = option.opt

# init log file
show_jot_opt(args)

# dataset
test_loader = data.DataLoader(create_dataset(args, 'test'), args.batch_size_test,
                              num_workers=args.num_workers, shuffle=False)
train_loader = data.DataLoader(create_dataset(args, 'train'), args.batch_size_train,
                               num_workers=args.num_workers, shuffle=True)

# init visualizer
visual = Visualizer(args)

# model
model = CapsNet(num_classes=train_loader.dataset.num_classes, opts=args)
print_log(model, args.file_name)

# optim
optimizer = []
if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay, momentum=args.momentum, )
elif args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay, betas=(args.beta1, 0.999))
elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum,
                              alpha=0.9, centered=True)
if args.scheduler is not None:
    scheduler = set_lr_schedule(optimizer, args.scheduler)
# loss
if args.loss_form == 'CE':
    criterion = nn.CrossEntropyLoss()
elif args.loss_form == 'spread':
    criterion = SpreadLoss(args, fix_m=args.fix_m,
                           num_classes=train_loader.dataset.num_classes)
elif args.loss_form == 'margin':
    # default loss
    criterion = MarginLoss(num_classes=train_loader.dataset.num_classes)
else:
    raise NameError('loss type not known')

# train and test
best_acc, best_epoch = 100, 0
epoch_size = len(train_loader)

for epoch in range(args.max_epoch):
# for epoch in range(1):

    t = time.time()
    old_lr = optimizer.param_groups[0]['lr']
    if epoch == 0:
        print_log('\ninit learning rate {:.10f} at iter {:d}\n'.format(
            old_lr, epoch), args.file_name)

    # TRAIN
    info = train(train_loader, model, criterion, optimizer, args, visual, epoch)
    # TEST
    if epoch >= args.show_test_after_epoch:
        extra_info = test(test_loader, model, criterion, args, visual, epoch)
    else:
        extra_info = dict()
        extra_info['test_loss'], extra_info['test_acc_error'], extra_info['test_acc5_error'] = 0, 100, 100

    # SHOW EPOCH SUMMARY
    info.update(extra_info)
    visual.print_loss(info, (epoch, 0, 0))

    # SAVE model
    test_acc = extra_info['test_acc_error']
    is_best = test_acc < best_acc
    best_epoch = epoch if is_best else best_epoch
    best_acc = min(test_acc, best_acc)
    save_checkpoint({
            'epoch':                epoch+1,
            'state_dict':           model.state_dict(),
            'test_acc_err':         test_acc,
            'best_test_acc_err':    best_acc,
            'optimizer':            optimizer.state_dict()},
        is_best, args, epoch)

    t_one_epoch = time.time() - t
    visual.print_info((epoch, epoch_size-1, epoch_size),
                      (True, old_lr, t_one_epoch/epoch_size, test_acc, best_acc, best_epoch))

    # ADJUST LR
    if args.scheduler is not None:
        scheduler.step(extra_info['test_acc_error'])
    else:
        adjust_learning_rate(optimizer, epoch, args)
    new_lr = optimizer.param_groups[0]['lr']
    if old_lr != new_lr:
        print_log('\nchange learning rate from {:.10f} to '
                  '{:.10f} at epoch {:d}\n'.format(old_lr, new_lr, epoch), args.file_name)

print_log('\nBest acc error: {:.4f} at epoch {:d}. Training done.'.format(best_acc, best_epoch), args.file_name)
visual.print_info((epoch, epoch_size-1, epoch_size),
                  (False, old_lr, t_one_epoch/epoch_size, test_acc, best_acc, best_epoch))










