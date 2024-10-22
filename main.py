import random
import torch
import torch.utils
import torch.utils.data
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

# Load CIFAR-10 dataset
def dataloader():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    return trainloader, testloader

def loadershow(loader):
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
    trainloader, testloader = dataloader()
    loadershow(trainloader)

if __name__ == "__main__":
    main()