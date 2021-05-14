# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


from tqdm import tqdm

import torch
import torch.nn as nn

import mlflow

from dataset import dataset_facory
from model import model_factory
from args import get_args
from optimizer import optimizer_factory, scheduler_factory


def val(model, criterion, optimizer, loader, device, iters, epoch):

    model.eval()
    loss_list = []
    acc_list = []
    with torch.no_grad(), tqdm(loader, leave=False) as pbar_loss:
        pbar_loss.set_description('[val]')
        for data, labels in pbar_loss:

            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            loss = criterion(output, labels)

            batch_loss = loss.item() / data.size(0)
            _, preds = output.max(1)
            batch_acc = preds.eq(labels).sum().item() / data.size(0)

            pbar_loss.set_postfix_str(
                'loss={:.05f}, acc={:.03f}'.format(batch_loss, batch_acc))

            loss_list.append(batch_loss)
            acc_list.append(batch_acc)

    val_loss = sum(loss_list) / len(loss_list)
    val_acc = sum(acc_list) / len(acc_list)

    mlflow.log_metrics({'val_loss': val_loss,
                        'val_acc': val_acc},
                       step=iters)


def train_classify(model, criterion, optimizer, loader, device, iters, epoch):

    model.train()

    with tqdm(loader, leave=False) as pbar_loss:
        pbar_loss.set_description('[train]')
        for data, labels in pbar_loss:

            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item() / data.size(0)
            _, preds = output.max(1)
            batch_acc = preds.eq(labels).sum().item() / data.size(0)

            pbar_loss.set_postfix_str(
                'loss={:.05f}, acc={:.03f}'.format(batch_loss, batch_acc))

            mlflow.log_metrics({'train_loss': batch_loss,
                                'train_acc': batch_acc},
                               step=iters)

            iters += 1

    return iters


def train(models, criterion, optimizers, loader, device, iters, epoch):

    net_G, net_D = models
    optimizer_G, optimizer_D = optimizers

    real_label_value = 1.
    fake_label_value = 0.

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    with tqdm(loader, leave=False) as pbar_loss:
        pbar_loss.set_description('[train]')
        for data, _ in pbar_loss:

            data = data.to(device)
            batch_size = data.size(0)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with all-real batch

            net_D.zero_grad()
            output_D = net_D(data).view(-1)

            real_label = torch.full((batch_size,), real_label_value,
                                    dtype=torch.float, device=device)
            loss_D_real = criterion(output_D, real_label)

            loss_D_real.backward()
            D_x = output_D.mean().item()

            # Train with all-fake batch

            z_noise = torch.randn(batch_size, nz, 1, 1, device=device)
            output_G = net_G(z_noise)
            output_D = net_D(output_G.detach()).view(-1)

            fake_label = torch.full((batch_size,), fake_label_value,
                                    dtype=torch.float, device=device)
            loss_D_fake = criterion(output_D, fake_label)
            loss_D_fake.backward()
            D_G_z1 = output_D.mean().item()

            loss_D = loss_D_real + loss_D_fake

            # Update D

            optimizer_D.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            net_G.zero_grad()
            output_D = net_D(output_G).view(-1)

            loss_G = criterion(output_D, real_label)

            loss_G.backward()
            optimizer_G.step()

            D_G_z2 = output_D.mean().item()

            pbar_loss.set_postfix_str(
                'L(D)={:.05f}, L(G)={:.05f},'
                'D(x)={:.05f}, D(G(z))={:.05f}/{:.05f}'.format(
                    loss_D.item(), loss_G.item(),
                    D_x, D_G_z1, D_G_z2))

            iters += 1

    return iters


def main():

    args = get_args()

    train_loader, val_loader, n_classes = dataset_facory(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    #
    #
    #
    #

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of training epochs
    num_epochs = 5

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    net_G, net_D = model_factory(args, n_classes)  # models = (G, D)
    net_G = net_G.to(device)
    net_D = net_D.to(device)
    net_G = nn.DataParallel(net_G)
    net_D = nn.DataParallel(net_D)
    models = [net_G, net_D]

    criterion = nn.BCELoss()
    optimizers = optimizer_factory(args, models)
    schedulers = scheduler_factory(args, optimizers)

    #
    #
    #
    #
    #

    iters = 0

    with tqdm(range(args.num_epochs)) as pbar_epoch:

        for epoch in pbar_epoch:
            pbar_epoch.set_description('[Epoch {}]'.format(epoch))

            iters = train(models, criterion, optimizers, train_loader, device,
                          iters, epoch)

            # if epoch % args.val_epochs:
            #     val(model, criterion, optimizer, val_loader, device,
            #         iters, epoch)

            if args.use_scheduler:
                for scheduler in schedulers:
                    scheduler.update()


if __name__ == "__main__":
    main()
