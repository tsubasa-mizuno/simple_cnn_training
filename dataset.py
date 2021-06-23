from torchvision import transforms
from torchvision.datasets import CIFAR10, UCF101
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
from torch.utils.data.dataloader import default_collate
import os


def transform_factory(args, do_crop=True):
    if do_crop:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
    return transform


def dataset_facory(args):
    """dataset factory

    Args:
        args (argparse): args

    Returns:
        DataLoader: training set loader
        DataLoader: validation set loader
    """

    transform = transform_factory(args)

    if args.dataset_name == "CIFAR10":
        train_set = CIFAR10(root=args.root,
                            train=True,
                            download=True,
                            transform=transform)
        val_set = CIFAR10(root=args.root,
                          train=False,
                          download=True,
                          transform=transform)
        n_classes = 10
        custom_collate = None

    elif args.dataset_name == "UCF101":

        # https://www.kaggle.com/pevogam/starter-ucf101-with-pytorch
        transform = transforms.Compose([
            # scale in [0, 1] of type float
            transforms.Lambda(lambda x: x / 255.),
            # reshape into (T, C, H, W) for easier convolutions
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            # rescale to the most common size
            transforms.Lambda(
                lambda x: nn.functional.interpolate(x, (240, 320))),
        ])

        def remove_audio_collate(batch):
            # https://www.kaggle.com/pevogam/starter-ucf101-with-pytorch
            '''
            remove audio channel because
            not all of UCF101 vidoes have audio channel
            '''
            video_only_batch = []
            for video, audio, label in batch:
                video_only_batch.append((video, label))
            return default_collate(video_only_batch)

        custom_collate = remove_audio_collate

        metadata_filename = os.path.join(
            args.metadata_path,
            'UCF101metadata_fpc{}_sbc{}.pickle'.format(
                args.frames_per_clip,
                args.step_between_clips))

        if not os.path.exists(metadata_filename):
            # precompute and save metadata
            dataset_dict = UCF101(root=args.root,
                                  annotation_path=args.annotation_path,
                                  frames_per_clip=args.frames_per_clip,
                                  step_between_clips=args.step_between_clips,
                                  num_workers=args.num_workers,
                                  )
            with open(metadata_filename, "wb") as f:
                pickle.dump(dataset_dict.metadata, f)

        with open(metadata_filename, 'rb') as f:
            metadata = pickle.load(f)

        train_set = UCF101(root=args.root,
                           annotation_path=args.annotation_path,
                           frames_per_clip=args.frames_per_clip,
                           step_between_clips=args.step_between_clips,
                           fold=1,
                           train=True,
                           transform=transform,
                           _precomputed_metadata=metadata)
        val_set = UCF101(root=args.root,
                         annotation_path=args.annotation_path,
                         frames_per_clip=args.frames_per_clip,
                         step_between_clips=args.step_between_clips,
                         fold=1,
                         train=False,
                         transform=transform,
                         _precomputed_metadata=metadata)
        n_classes = 101
    else:
        raise ValueError("invalid args.dataset_name")

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              collate_fn=custom_collate,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=custom_collate,
                            num_workers=args.num_workers)

    return train_loader, val_loader, n_classes
