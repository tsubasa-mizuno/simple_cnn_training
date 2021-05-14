import argparse
import mlflow


def get_args():
    '''generate argparse object

    Returns:
        args: [description]
    '''
    parser = argparse.ArgumentParser(description='simple CNN model')

    # dataset
    parser.add_argument('-r', '--root', type=str, default='./data',
                        help='root of dataset. default to ./data')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10',
                        help='name of dataset. default to CIFAR10')
    parser.add_argument('-s', '--img_size', type=int, default=224,
                        help='image size. default to 224')
    # model
    parser.add_argument('--torch_home', type=str, default='./models',
                        help='TORCH_HOME where pre-trained models are stored.'
                        ' default to ./models')
    parser.add_argument('-m', '--model', type=str, default='resnet18',
                        help='CNN model. default to resnet18')
    parser.add_argument('--pretrain', dest='pretrain', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false',
                        help='do not use pretrained model')
    parser.set_defaults(pretrain=True)

    # training
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='batch size. default to 8')
    parser.add_argument('-w', '--num_workers', type=int, default=2,
                        help='number of workers. default to 2')
    parser.add_argument('-e', '--num_epochs', type=int, default=25,
                        help='number of epochs. default to 25')
    parser.add_argument('--val_epochs', type=int, default=2,
                        help='validation interval in epochs. default to 2')
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer. SGD or Adam. default to SGD')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='learning rate. default to 0.0001')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of SGD. default to 0.9')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999],
                        help='betas of Adam. default to (0.9, 0.999).'
                        'specify like --betas 0.9 0.999')
    parser.add_argument('--use_scheduler', dest='use_scheduler',
                        action='store_true',
                        help='use scheduler')
    parser.set_defaults(use_scheduler=False)

    args = parser.parse_args()
    print(args)

    mlflow.log_param('dataset', args.dataset_name)
    mlflow.log_param('model', args.model)
    mlflow.log_param('use pretrain', args.pretrain)
    mlflow.log_param('optimizer', args.optimizer)
    mlflow.log_param('learning_rate', args.lr)
    mlflow.log_param('momentum', args.momentum)
    mlflow.log_param('batch_size', args.batch_size)

    return args
