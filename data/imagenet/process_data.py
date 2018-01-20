import scipy.io as sp
import h5py
import numpy as np
import os
from scipy.misc import imread, imsave
from object_detection.utils.util import mkdirs


train_bbox_gt = '/media/hongyang/research_at_large/Q-dataset/imagenet_cls/tidy_up_1k'
train_ori_im = '/home/hongyang/dataset/imagenet_cls/cls/train'
val_bbox_gt = '/home/hongyang/dataset/imagenet_cls/val_gt_bbox.mat'
val_ori_im = '/home/hongyang/dataset/imagenet_cls/cls/val'
cls_num = 1000

save_prefix = 'set1'    # differnt settings
val_folder = os.path.join(os.getcwd(), save_prefix, 'val')
train_folder = os.path.join(os.getcwd(), save_prefix, 'train')
start_cls = 1
end_cls = 1000
phase = 'val'


def read_hdf5_ref(file, ref):
    obj = file[ref]
    string = ''.join(chr(i) for i in obj[:])
    return string


def sort_up_file(file, raw_list):
    file_name = os.path.join(os.getcwd(), 'val_im_list.npy')
    if os.path.exists(file_name):
        im_list = np.load(file_name)
    else:
        im_list = np.empty([len(raw_list), 1], dtype=object)
        # for i in range(10):
        for i in range(len(raw_list)):
            im_list[i] = read_hdf5_ref(file, raw_list[i][0])

        np.save(file_name, im_list)
    return im_list


def crop_and_save(img, curr_bbox, im_path, setting):
    inst_num = curr_bbox.shape[0]
    curr_bbox = curr_bbox.astype(int)
    for i in range(inst_num):
        curr_inst = img[curr_bbox[i, 1]:curr_bbox[i, 3], curr_bbox[i, 0]:curr_bbox[i, 2]]
        inst_path = im_path[:-5] + '_{:d}'.format(i+1) + '.JPEG'
        # if curr_inst is []:
        #     continue
        # else:
        try:
            imsave(inst_path, curr_inst)
        except:
            # in this case, curr_inst == []
            continue


def _process(file, ref_list):
    result = list()
    for i in range(len(ref_list)):
        result.append(''.join(chr(j) for j in file[ref_list[i][0]]))
    return result


def _process2(file, ref_list):
    result = list()
    for i in range(len(ref_list)):
        result.append(np.asarray(file[ref_list[i][0]].value).transpose())
    return result


## VAL
## val_gt = sp.loadmat(val_bbox_gt)
if phase == 'val':
    val_gt = {}
    f = h5py.File(val_bbox_gt)
    for k, v in f.items():
        val_gt[k] = np.array(v)
    val_bbox_list = val_gt['bbox'].transpose()
    val_im_list = sort_up_file(f, val_gt['image_list'].transpose())

    cnt = 0
    for root, dirs, files in sorted(os.walk(val_ori_im)):

        if files != []:
            # the current class
            cnt += 1
            if start_cls <= cnt <= end_cls:
                curr_cls_name = os.path.basename(root)
                curr_cls_folder = os.path.join(val_folder, curr_cls_name)
                mkdirs(curr_cls_folder)

                for im in files:
                    index = [i for i, name in enumerate(val_im_list) if name == im]
                    curr_bbox = val_bbox_list[index, :]
                    if curr_bbox != []:
                        img = imread(os.path.join(root, im))
                        im_path = os.path.join(curr_cls_folder, im)
                        crop_and_save(img, curr_bbox, im_path, save_prefix)

                print('processed VAL cls_id: {:04d}, range({:04d}:{:04d})...'.format(
                    cnt, start_cls, end_cls))


## TRAIN
# note: f[f['bbox_new']['REVISED'][index][0]]
if phase == 'train':
    cnt = 0
    for root, dirs, files in sorted(os.walk(train_bbox_gt)):
        for curr_cls in files:
            if curr_cls[-3:] == 'mat':
                cnt += 1

                if start_cls <= cnt <= end_cls:
                    curr_cls_name = curr_cls[:-4]
                    curr_cls_save_path = os.path.join(train_folder, curr_cls_name)
                    mkdirs(curr_cls_save_path)

                    try:
                        f = h5py.File(os.path.join(root, curr_cls))
                    except:
                        # for unknown reason, some mat files cant be opened
                        print('WARNING: cls {:s}, cnt={:d} is skipped!'.format(curr_cls_name, cnt))
                        continue
                    already_saved_num = len(os.listdir(curr_cls_save_path))
                    if already_saved_num >= len(f['bbox_new']['xml']):
                        print('HEADS UP! cls {:s}, cnt={:d} is already saved!'.format(curr_cls_name, cnt))
                        f.close()
                        continue

                    im_bbox_xml = _process2(f, f['bbox_new']['xml'])
                    im_bbox = _process2(f, f['bbox_new']['bbox'])
                    im_list = _process(f, f['bbox_new']['im_name'])
                    f.close()
                    for ind, curr_im in enumerate(im_list):
                        curr_bbox = im_bbox[ind] if im_bbox_xml[ind].ndim == 1 else im_bbox_xml[ind]
                        if curr_bbox.ndim == 1:
                            continue
                        else:
                            crop_and_save(imread(os.path.join(train_ori_im, curr_cls_name, curr_im)),
                                          curr_bbox,
                                          os.path.join(curr_cls_save_path, curr_im),
                                          save_prefix)
                    print('processed TRAIN cls_id: {:04d}, range({:04d}:{:04d})...'.format(
                        cnt, start_cls, end_cls))

