import torch.optim as optim
from torch.optim import lr_scheduler


def optimizer_factory(args, models):
    optimizer = []
    if args.optimizer == 'SGD':
        for model in models:
            optimizer.append(optim.SGD(model.parameters(),
                                       lr=args.lr, momentum=args.momentum))
    elif args.optimizer == 'Adam':
        for model in models:
            optimizer.append(optim.Adam(model.parameters(),
                                        lr=args.lr, betas=args.betas))
    else:
        raise ValueError("invalid args.optimizer")

    return optimizer


def scheduler_factory(args, optimizers):
    scheduler = []
    if args.use_scheduler:
        for optimizer in optimizers:
            scheduler.append(lr_scheduler.StepLR(
                optimizer, step_size=7, gamma=0.1))
    else:
        scheduler = None
    return scheduler
