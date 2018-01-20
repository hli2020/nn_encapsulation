import scipy.io as sp
import h5py
from scipy.misc import imread, imsave
from object_detection.utils.util import *

cls_num = 1000
save_prefix = 'set1'    # differnt settings
val_folder = os.path.join(os.getcwd(), save_prefix, 'val')
train_folder = os.path.join(os.getcwd(), save_prefix, 'train')
start_cls = 1
end_cls = 1000
phase = 'val_train'

file_name = 'stats_per_cls_{:s}.txt'.format(phase)
#TODO: compute area

cnt = 0
print_log('cls_index\tori_num\ttidy_up_num(w and h>=50)\ttrain_ori_num\ttidy_up_num', file_name, init=True)
for root, dirs, files in sorted(os.walk(val_folder)):

    if files != []:
        # the current class
        cnt += 1
        if start_cls <= cnt <= end_cls:

            train_files = os.listdir(root.replace('val', 'train'))

            im_h_w = np.array([imread(os.path.join(root, x)).shape[0:2] for x in files])
            index = (im_h_w[:, 0] >= 50) & (im_h_w[:, 1] >= 50)

            train_im_h_w = np.array([imread(os.path.join(root.replace('val', 'train'), x)).shape[0:2]
                                     for x in train_files])
            train_index = (train_im_h_w[:, 0] >= 50) & (train_im_h_w[:, 1] >= 50)
            print_log('{:04d}\t{:04d}\t{:04d}\t{:04d}\t{:04d}'.format(cnt, im_h_w.shape[0], sum(index),
                                                                  train_im_h_w.shape[0], sum(train_index)), file_name)


