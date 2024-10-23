import random
import argparse
import torch
import torch.utils
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from model import VisionTransformer

# CIFAR-10 classes name
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Ensure reproducibility
def set_seed(num: int):
    torch.manual_seed(num)
    random.seed(num)
    np.random.seed(num)

def hyperparameters():
    parser = argparse.ArgumentParser()
    
    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument('-lr', "--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])

    # Data Arguments
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--n_classes", type=int, default=10)

    # ViT Arguments
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_attention_heads", type=int, default=4)
    parser.add_argument("--forward_mul", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--dropout", type=int, default=0.1)

    args = parser.parse_args()
    return args

# Load CIFAR-10 dataset
def dataloader(batch_size: int,  num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers,
                                            pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers,
                                            pin_memory=True)
    return trainloader, testloader

def loadershow(loader: DataLoader):
    # get some random images from dataloader
    dataiter = iter(loader)
    images, labels = next(dataiter)

    # show images shape
    print(images.shape) # (b, c, h, w)

    # show labels
    print(' '.join(f'{classes[labels[j]]}' for j in range(4)))

    # show image
    grid_images = torchvision.utils.make_grid(images)
    grid_images = (grid_images / 2 + 0.5).numpy() # unnormalize
    plt.imshow(np.transpose(grid_images, (1, 2, 0))) # (c, h, w) -> (h, w, c)
    plt.show()

def train(args, trainloader: DataLoader, model: nn.Module):
    len_trainloader = len(trainloader)

    optimizer = optim.AdamW(model.parameters(), args.learning_rate, weight_decay=1e-3)

    # scheduler for linear warmup of lr and then cosine decay to 1e-5
    linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/args.warmup_epochs, 
                                                end_factor=1.0, total_iters=args.warmup_epochs-1,
                                                last_epoch=-1, verbose=True)
    cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.warmup_epochs,
                                                     eta_min=1e-5, verbose=True)
    
    # loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # variable to capture best test accuracy
    best_acc = 0

    # training loop
    for epoch in tqdm(range(args.epochs)):

        # set model to training mode
        model.train()

        # put model to device
        model = model.to(args.device)

        # arrays to record epoch loss and accuracy
        train_epoch_loss = []
        train_epoch_accuracy = []

        # loop in loader
        for i, (x, y) in enumerate(trainloader):
            # put data to device
            x, y = x.to(args.device), y.to(args.device)

            # get output logits from the model
            logits = model(x)

            # computer training loss
            loss = loss_fn(logits, y)

            # batch matrix
            batch_pred = logits.max(1)[1]
            batch_accuracy = (y==batch_pred).float().mean()
            train_epoch_loss += [loss.item()]
            train_epoch_accuracy += [batch_accuracy.item()]

            # update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # update learning rate using schedulers
        if epoch < args.warmup_epochs:
            linear_warmup.step()
        else:
            cos_decay.step()

        print(f"loss: {loss.item():.4f}  |  accuracy: {batch_accuracy.item():.4f}")

        # save model every 20 epochs
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"./output/ViT_model_{epoch:0>3}.pt")

def main():
    set_seed(1234)
    args = hyperparameters()
    trainloader, testloader = dataloader(args.batch_size, args.num_workers)

    # loadershow(trainloader)
    model = VisionTransformer(args.n_channels, args.embed_dim, args.n_layers, 
                              args.n_attention_heads, args.forward_mul, args.image_size, 
                              args.patch_size, args.n_classes, args.dropout)
    
    # print(model)
    train(args, trainloader, model)

if __name__ == "__main__":
    main()