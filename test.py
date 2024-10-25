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
    
    # Test Arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output_path", type=str, default='./output')
    parser.add_argument("--timestamp", type=str, default="1900-01-01-00-00")

    # Data Arguments
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--data_path", type=str, default='./data')
    parser.add_argument("--num_test_images", type=int, default=None)
    parser.add_argument("--index", type=int, default=1)

    # ViT Arguments
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_attention_heads", type=int, default=4)
    parser.add_argument("--forward_mul", type=int, default=2)
    parser.add_argument("--dropout", type=int, default=0.1)
    parser.add_argument("--model_path", type=str, default='model/vit-classification-cifar10-colab-t4/ViT_model_199.pt')

    args = parser.parse_args()
    return args

# Load CIFAR-10 dataset
def dataloader(args: argparse.ArgumentParser) -> DataLoader:
    test_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                            download=True, transform=test_transform)
    
    if args.num_test_images != None:
        test_subset = Subset(testset, torch.arange(args.num_test_images))
    else:
        test_subset = testset
    
    testloader = DataLoader(test_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    
    print(f"Test num: {len(test_subset)}")

    return testloader

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
    plt.axis('off')
    plt.show()

def load_single_image(args: argparse.ArgumentParser, loader: DataLoader):
    assert args.index < args.batch_size

    dataiter = iter(loader)
    images, labels = next(dataiter)
    image = images[args.index]
    label = labels[args.index]

    return image, label

def show_single_image(image:torch.Tensor, label: str):
    image = (image / 2 + 0.5).numpy()
    plt.imshow(np.transpose(image, (1, 2, 0))) # (c, h, w) -> (h, w, c)
    plt.title(label)
    plt.axis('off')
    plt.show()
    

def test(args: argparse.ArgumentParser, testloader: DataLoader, model: nn.Module) -> list:
    """
    test model with dataloader
    """
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

def test_single(args: argparse.ArgumentParser, image: torch.Tensor, model: nn.Module):
    """
    test model with single image
    """
    # set model to evaluation mode
    model.eval()

    # put model, image to device
    model = model.to('cpu')
    image = image.to('cpu')

    with torch.no_grad():
        logit = model(image)
    
    return logit

def evaluate_cifar(args: argparse.ArgumentParser, loader: DataLoader, model: nn.Module):
    """
    evaluate whole CIFAR 10000 test images
    """
    test(args, loader, model)

def evaluate_single(args: argparse.ArgumentParser, loader: DataLoader, model: nn.Module):
    """
    evaluate single image in CIFAR test set
    """
    image, label = load_single_image(args, loader)
    output = test_single(args, image.unsqueeze(dim=0), model)

    label = label.numpy()
    output = output.max(1)[1].numpy()[0]

    if label == output:
        print(f"<Correct> {classes[label]}(label) {classes[output]}(output)")
    else:
        print(f"<Wrong> - {classes[label]}(label) {classes[output]}(output)")

    show_single_image(image, classes[output])

def main():
    set_seed(1234)
    args = hyperparameters()
    testloader = dataloader(args)

    model = VisionTransformer(args.n_channels, args.embed_dim, args.n_layers, 
                              args.n_attention_heads, args.forward_mul, args.image_size, 
                              args.patch_size, args.n_classes, args.dropout)
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=args.device))
    
    evaluate_single(args, testloader, model)

if __name__ == "__main__":
    main()