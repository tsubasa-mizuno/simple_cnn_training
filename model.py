import os

import torch.nn as nn
from torchvision.models import resnet18

from model_dcgan import Generator, Discriminator, weights_init


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
        models = (Generator(), Discriminator())
        for model in models:
            model.apply(weights_init)
        model = models

    else:
        raise ValueError("invalid args.model")

    return model
