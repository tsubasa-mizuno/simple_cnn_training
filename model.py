import os

import torch.nn as nn
from torchvision.models import resnet18

from model_dcgan import Generator, Discriminator


def model_factory(args, n_classes):

    if args.pretrain:
        # Specity the directory where a pre-trained model is stored.
        # Otherwise, by default, models are stored in users home dir `~/.torch`
        os.environ['TORCH_HOME'] = args.torch_home

    if args.model == 'resnet18':
        model = resnet18(pretrained=args.pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)

    elif args.model == 'DCGAN':
        model = (Generator(), Discriminator())

    else:
        raise ValueError("invalid args.model")

    return model
