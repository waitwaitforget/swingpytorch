import torch as t
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np 


def get_dataloaders(config):
    """
    Create dataloaders according to configuration.
    config:
        config: basic config obj or argparse obj
    return:
        trainloader, valloader
    """
    # Data
    print('==> Preparing dataset %s' % config.dataset)
    if config.no_aug:
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def _init_fn(worker_id):
        np.random.seed(config.manualSeed)

    if config.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        #mydataset = myds.CIFAR10
        num_classes = 10
        config.K = 2
    else:
        dataloader = datasets.CIFAR100
        #mydataset = myds.CIFAR100
        num_classes = 100

    trainset = dataloader(root=config.data_dir, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=config.train_batch, shuffle=True, num_workers=config.workers,
        worker_init_fn=_init_fn)

    testset = dataloader(root=config.data_dir, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
        worker_init_fn=_init_fn)

    return trainloader, testloader