"""VOC Dataset Classes and COCO
Updated by: Ellis Brown, Max deGroot, Hongyang Li

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
"""

import os
import os.path
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from utils.util import ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


def detection_collate(batch):
    """Custom collate function for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False, coco_cls=None):

        classes = VOC_CLASSES if coco_cls is None else coco_cls
        self.coco_cls = coco_cls
        self.class_to_ind = class_to_ind or dict(zip(classes, range(len(classes))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        if self.coco_cls is None:
            for obj in target.iter('object'):
                difficult = int(obj.find('difficult').text) == 1
                if not self.keep_difficult and difficult:
                    continue
                name = obj.find('name').text.lower().strip()
                bbox = obj.find('bndbox')

                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(bbox.find(pt).text) - 1
                    # scale height or width
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                    bndbox.append(cur_pt)
                label_idx = self.class_to_ind[name]
                bndbox.append(label_idx)
                res += [bndbox]
                # img_id = target.find('filename').text[:-4]
        else:
            for i in range(len(target)):
                curr_bbox = target[i]['bbox']
                curr_bbox = [curr_bbox[0]/width, curr_bbox[1]/height,
                             (curr_bbox[2]+curr_bbox[0])/width,
                             (curr_bbox[3]+curr_bbox[1])/height]
                lable_idx = self.class_to_ind[target[i]['category_id']]
                curr_bbox.append(lable_idx)
                res += [curr_bbox]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, transform=None, target_transform=None, dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.num_classes = 20 + 1
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            # target = [obj_num x 5], each coor [0,1] and last colum [0:cls_num-1]
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        # DEPRECATED
        """Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        """
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        # DEPRECATED
        """Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        """
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        # DEPRECATED
        """Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


class COCODetection(data.Dataset):

    def __init__(self, root, phase,
                 transform=None, dataset_name='COCO'):
        from pycocotools.coco import COCO
        if phase == 'train':
            anno_file = 'instances_train2014.json'
            anno_file_2 = 'instances_valminusminival2014.json'
            anno_file_2 = (root + '/annotations/' + anno_file_2)
            self.coco_2 = COCO(anno_file_2)
            self.im_path = root + '/train2014'
            self.im_path_2 = root + '/val2014'
        else:
            anno_file = 'instances_minival2014.json'
            self.im_path = root + '/val2014'

        anno_file = (root + '/annotations/' + anno_file)
        self.coco = COCO(anno_file)
        self.ids = list(self.coco.imgs.keys())
        if phase == 'train':
            self.ids.extend(list(self.coco_2.imgs.keys()))
        self.COCO_CLASSES = [v['id'] for k, v in self.coco.cats.items()]
        self.COCO_CLASSES_names = [v['name'] for k, v in self.coco.cats.items()]
        self.transform = transform
        self.target_transform = AnnotationTransform(coco_cls=self.COCO_CLASSES)
        self.name = dataset_name
        self.num_classes = len(self.COCO_CLASSES) + 1
        self.phase = phase

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        coco = self.coco
        if self.phase == 'train':
            coco_2 = self.coco_2

        valid_im = False
        while not valid_im:
            img_id = self.ids[index]
            try:
                path = coco.loadImgs(img_id)[0]['file_name']  # will report an error
                ann_ids = coco.getAnnIds(imgIds=img_id)
                target = coco.loadAnns(ann_ids)
            except KeyError:
                # print('this train image is from val')
                path = coco_2.loadImgs(img_id)[0]['file_name']
                ann_ids = coco_2.getAnnIds(imgIds=img_id)
                target = coco_2.loadAnns(ann_ids)
            if len(target) == 0:
                index += 1
                # print('im: {:s} no annotation!'.format(path))
            else:
                valid_im = True

        # img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = cv2.imread(os.path.join(self.im_path, path))
        if img is None:
            img = cv2.imread(os.path.join(self.im_path_2, path))
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            # try:
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # except:
            #     print(target.shape)
            #     print(target.ndim)
            #     raise
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
