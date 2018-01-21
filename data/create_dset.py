import torchvision.datasets as dset
import torchvision.transforms as T
import torch
from data.imagenet import ImageNet


def create_dataset(opts, phase=None):
    name = opts.dataset

    if phase == 'train':
        # T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # T.Resize(40),
        # T.RandomCrop(32),
        # T.RandomHorizontalFlip(),
        # T.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
        transform = T.Compose([
            T.Resize(40),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
            T.ToTensor(),
        ])
    elif phase == 'test':
        if opts.multi_crop_test:
            transform = T.Compose([
                T.Resize(40),
                T.TenCrop(32),
                T.Lambda(lambda crops:
                         torch.stack([T.ToTensor()(crop) for crop in crops]))  # returns a 4D tensor
            ])
        else:
            transform = T.Compose([
                T.Resize(40),
                T.RandomCrop(32),
                T.ToTensor(),
            ])
    if name == 'cifar10':
        dataset = dset.CIFAR10(root='data', train=phase == 'train',
                               transform=transform, download=True)
        dataset.num_classes = 10
    elif name == 'cifar100':
        dataset = dset.CIFAR100(root='data', train=phase == 'train',
                                transform=transform, download=True)
        dataset.num_classes = 100
    elif name == 'svhn':
        split_name = 'train' if phase == 'train' else 'test'
        dataset = dset.SVHN(root='data', split=split_name,
                            transform=transform, download=True)
        dataset.num_classes = 10
    elif name == 'fmnist':
        dataset = dset.FashionMNIST(root='data/fmnist', train=phase == 'train',
                                    transform=transform, download=True)
        dataset.num_classes = 10

    elif name == 'tiny_imagenet':

        if phase == 'test':
            phase = 'val'
        root_name = 'data/imagenet/set1/' + phase

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        if opts.bigger_input:
            resize_size, crop_size = 256, 224
        else:
            resize_size, crop_size = 156, 128
        if phase == 'train':
            transform = T.Compose([
                T.Resize(resize_size),
                T.CenterCrop(crop_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
                T.ToTensor(),
                normalize
            ])
        elif phase == 'val':
            transform = T.Compose([
                T.Resize(resize_size),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                normalize
            ])

        if hasattr(opts, 'setting') is None:
            raise(RuntimeError("Setting is none in the imagenet case! Are you fucking me?"))

        dataset = ImageNet(root_name, opts.setting, transform)
        dataset.num_classes = 150 if opts.setting == 'obj_det' else 200

    else:
        raise NameError('Unknown dataset')

    dataset.name = name
    print('{:s} on {:s}'.format(phase.upper(), dataset.name))

    return dataset
