import torchvision.datasets as dset
import torchvision.transforms as T
import torch


def create_dataset(opts, phase=None):
    name = opts.dataset

    if phase == 'train':
        # T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
    else:
        raise NameError('Unknown dataset')

    dataset.name = name
    print('{:s} on {:s}'.format(phase.upper(), dataset.name))

    return dataset
