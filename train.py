import os
import datetime
import random
import argparse
import torch
import torch.utils
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
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
    parser.add_argument("--output_path", type=str, default='./output')
    parser.add_argument("--timestamp", type=str, default="1900-01-01-00-00")

    # Data Arguments
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--data_path", type=str, default='./data')
    parser.add_argument("--num_train_images", type=int, default=None)
    parser.add_argument("--num_test_images", type=int, default=None)

    # ViT Arguments
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_attention_heads", type=int, default=4)
    parser.add_argument("--forward_mul", type=int, default=2)
    parser.add_argument("--dropout", type=int, default=0.1)
    parser.add_argument("--model_path", type=str, default='./model')

    args = parser.parse_args()
    return args

# Load CIFAR-10 dataset
def dataloader(args: argparse.ArgumentParser) -> DataLoader:
    train_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.RandomCrop(args.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                            download=True, transform=test_transform)
    
    if args.num_train_images != None:
        train_subset = Subset(trainset, torch.arange(args.num_train_images))
    else:
        train_subset = trainset
    
    if args.num_test_images != None:
        test_subset = Subset(testset, torch.arange(args.num_test_images))
    else:
        test_subset = testset
    
    trainloader = DataLoader(train_subset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)
    testloader = DataLoader(test_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    
    print(f"Train num: {len(train_subset)}\nTest num: {len(test_subset)}")

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

def train(args: argparse.ArgumentParser, model: nn.Module,
          trainloader: DataLoader, testloader: DataLoader) -> list:
    iters_per_epoch = len(trainloader)

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

    # arrays to record training progression
    train_losses     = []
    test_losses      = []
    train_accuracies = []
    test_accuracies  = []

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
            logits, att_mat_full = model(x)

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

            # Log training progress
            if i % 50 == 0 or i == (iters_per_epoch - 1):
                print(f'Ep: {epoch+1}/{args.epochs}\tIt: {i+1}/{iters_per_epoch}\tbatch_loss: {loss:.4f}\tbatch_accuracy: {batch_accuracy:.2%}')
        
        # test testing set every epoch
        test_loss, test_acc, _ = test(args, testloader, model)

        # Capture best test accuracy
        best_acc = max(test_acc, best_acc)
        print(f"Best test acc: {best_acc:.2%}\n")

        # save model
        torch.save(model.state_dict(), f"{args.model_path}/{args.timestamp}/ViT_model_{epoch:0>3}.pt")

        # update learning rate using schedulers
        if epoch < args.warmup_epochs:
            linear_warmup.step()
        else:
            cos_decay.step()

        # Update training progression metric arrays
        train_losses += [sum(train_epoch_loss)/iters_per_epoch]
        test_losses += [test_loss]
        train_accuracies += [sum(train_epoch_accuracy)/iters_per_epoch]
        test_accuracies += [test_acc]
    
    return train_losses, train_accuracies, test_losses, test_accuracies

def test(args:argparse.ArgumentParser, testloader: DataLoader, model: nn.Module) -> list:
    # set model to evaluation mode
    model.eval()

    # put model to device
    model = model.to(args.device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # arrays to record all labels and logits
    all_labels = []
    all_logits = []

    for (x, y) in testloader:
        # put data to device
        x = x.to(args.device)

        # avoid capturing gradients in evaluation time for faster speed
        with torch.no_grad():
            logits = model(x)

        all_labels.append(y)
        all_logits.append(logits.cpu())

     # convert all captured variables to torch
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)
    all_pred = all_logits.max(1)[1]

    # Compute loss, accuracy and confusion matrix
    loss = loss_fn(all_logits, all_labels).item()
    acc = accuracy_score(y_true=all_labels, y_pred=all_pred)
    cm = confusion_matrix(y_true=all_labels, y_pred=all_pred, labels=range(args.n_classes))

    print(f"Test acc: {acc:.2%}\tTest loss: {loss:.4f}\nTest Confusion Matrix:")
    print(cm)

    return loss, acc, cm

def plot_graphs(args: argparse.ArgumentParser,
                train_losses: list, train_accuracies: list, 
                test_losses: list, test_accuracies: list):
    # Plot graph of loss values
    plt.plot(train_losses, color='b', label='Train')
    plt.plot(test_losses, color='r', label='Test')

    plt.ylabel('Loss', fontsize = 18)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize = 18)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=15, frameon=False)

    # plt.show()  # Uncomment to display graph
    plt.savefig((f'{args.output_path}/{args.timestamp}/graph_loss.png'), bbox_inches='tight')
    plt.close('all')

    # Plot graph of accuracies
    plt.plot(train_accuracies, color='b', label='Train')
    plt.plot(test_accuracies, color='r', label='Test')

    plt.ylabel('Accuracy', fontsize = 18)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize = 18)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=15, frameon=False)

    # plt.show()  # Uncomment to display graph
    plt.savefig((f'{args.output_path}/{args.timestamp}/graph_accuracy.png'), bbox_inches='tight')
    plt.close('all')

def main():
    set_seed(1234)
    args = hyperparameters()

    time = datetime.datetime.now()
    args.timestamp = str(time.strftime('%Y-%m-%d-%H-%M'))

    # Create required directories if they don't exist
    os.makedirs(f'{args.model_path}/{args.timestamp}',  exist_ok=True)
    os.makedirs(f'{args.output_path}/{args.timestamp}', exist_ok=True)

    trainloader, testloader = dataloader(args)

    # loadershow(trainloader)
    model = VisionTransformer(args.n_channels, args.embed_dim, args.n_layers, 
                              args.n_attention_heads, args.forward_mul, args.image_size, 
                              args.patch_size, args.n_classes, args.dropout)
    
    # print(model)
    train_losses, train_accuracies, test_losses, test_accuracies = train(args, model, trainloader, testloader)
    plot_graphs(args, train_losses, train_accuracies, test_losses, test_accuracies)

if __name__ == "__main__":
    main()