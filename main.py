import random
import argparse
import torch
import torch.utils
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

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
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    return args

# Load CIFAR-10 dataset
def dataloader(train_batch_size: int, 
               test_batch_size: int, 
               num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=train_batch_size,
                                            shuffle=True, num_workers=num_workers,
                                            pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=test_batch_size,
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

def main():
    set_seed(1234)
    args = hyperparameters()
    trainloader, testloader = dataloader(args.train_batch_size,
                                         args.test_batch_size, 
                                         args.num_workers)
    loadershow(trainloader)

if __name__ == "__main__":
    main()