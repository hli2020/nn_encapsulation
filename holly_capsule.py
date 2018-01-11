from __future__ import print_function
import torch.optim as optim
import torch.utils.data as data
from data.create_dset import create_dataset
from layers.capsule import CapsNet
from layers.cap_layer import MarginLoss, SpreadLoss
from layers.train_val import *

from object_detection.utils.visualizer import Visualizer
from object_detection.utils.util import *
from object_detection.utils.train import set_lr_schedule, adjust_learning_rate
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
best_acc, best_epoch = 0, 0
for epoch in range(args.max_epoch):

    old_lr = optimizer.param_groups[0]['lr']
    if epoch == 0:
        print_log('\ninit learning rate {:f} at iter {:d}\n'.format(
            old_lr, epoch), args.file_name)

    # TRAIN
    info = train(train_loader, model, criterion, optimizer, args, visual, epoch)
    # TEST
    if epoch >= args.show_test_after_epoch:
        extra_info = test(test_loader, model, criterion, args, visual, epoch)
    else:
        extra_info = dict()
        extra_info['test_loss'], extra_info['test_acc'] = 0, 0

    # SHOW EPOCH SUMMARY
    info.update(extra_info)
    visual.print_loss(info, epoch, epoch_sum=True)

    # ADJUST LR
    if args.scheduler is not None:
        scheduler.step(extra_info['test_acc'])
    else:
        adjust_learning_rate(optimizer, epoch, args)
    new_lr = optimizer.param_groups[0]['lr']
    if old_lr != new_lr:
        print_log('\nchange learning rate from {:f} to '
                  '{:f} at iter {:d}\n'.format(old_lr, new_lr, epoch), args.file_name)

    # SAVE model
    test_acc = 0 if extra_info['test_acc'] == 'n/a' else extra_info['test_acc']
    is_best = test_acc > best_acc
    best_epoch = epoch if is_best else best_epoch
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
            'epoch':            epoch+1,
            'state_dict':       model.state_dict(),
            'test_acc':         test_acc,
            'best_test_acc':    best_acc,
            'optimizer':        optimizer.state_dict(),
    }, is_best, args, epoch)
    if args.use_KL:
        kl_info = '<br/>KL_factor: {:.4f}'.format(args.KL_factor)
    else:
        kl_info = ''
    common_suffix = 'curr best test acc {:.4f} at epoch {:d}<br/>' \
                    'curr lr {:f}<br/>' \
                    'epoch [{:d} | {:d}]{:s}'.format(
                        best_acc, best_epoch, new_lr, epoch,
                        args.epochs, kl_info)
    msg = 'status: <b>RUNNING</b><br/>' + common_suffix
    visual.vis.text(msg, win=200)

print_log('Best acc: {:.4f}. Training done.'.format(best_acc), args.file_name)
msg = 'status: <b>DONE</b><br/>' + common_suffix
visual.vis.text(msg, win=200)










