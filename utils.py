import torch.cuda
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import yaml
import logging
from easydict import EasyDict as edict
import pickle

from model import GoogLeNet, ResNet,VGG


def get_transform(mode="train"):
    if mode == "train":
        train_tfm = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return train_tfm

    elif mode == "test":
        test_tfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ])
        return test_tfm


def get_dataloader(batch_size):
    train_set = CIFAR100(root='./', train=True, download=True, transform=get_transform("train"))
    test_set = CIFAR100(root='./', train=False, download=True, transform=get_transform("test"))

    validation_split = 0.1
    val_size = int(validation_split * len(train_set))
    train_size = len(train_set) - val_size
    train_set, valid_set = random_split(train_set, [train_size, val_size], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size//2, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader


def load_config(file):
    with open(file) as f:
        config = edict(yaml.load(f.read(), Loader=yaml.FullLoader))
        config.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config


def get_model(config):
    model = config.model
    if model == "google":
        model = GoogLeNet(num_classes=config.num_classes,
                          aux_logits=config.aux_logits,
                          init_weights=config.init_weights)
    elif model == "vgg":
        model = VGG(config.num_classes)
    elif model == "resnet":
        model = ResNet(config.num_classes)
    return model

def dump_data(config,*data):
    with open(config.save_data,"wb") as f:
        pickle.dump([*data],f,protocol=pickle.HIGHEST_PROTOCOL)
