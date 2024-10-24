# ViT-Classification-CIFAR10

This repository contains an implementation of the **Vision Transformer (ViT)** from scratch using PyTorch. The model is applied to the CIFAR-10 dataset for image classification. Vision Transformers divide an image into smaller patches and process them with transformer layers to extract features, leading to state-of-the-art performance on various vision tasks.

## Features

- Implementation of Vision Transformer from scratch.
- Trains and evaluates on CIFAR-10 dataset.
- Supports adjustable hyperparameters like patch size, learning rate, and more.
- Includes learning rate warmup and weight initialization strategies.
- Can run on CPU, CUDA, or MPS (for Apple Silicon).

## Table of Contents

- [ViT-Classification-CIFAR10](#vit-classification-cifar10)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Model Architecture](#model-architecture)
  - [Results](#results)
  - [References](#references)
  - [License](#license)

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/nick8592/ViT-Classification-CIFAR10.git
cd ViT-Classification-CIFAR10
pip install -r requirements.txt
```

## Usage

To train the Vision Transformer on the CIFAR-10 dataset, you can run the following command:

```bash
python train.py --batch_size 128 --epochs 200 --learning_rate 0.0005
```

## Arguments

The following arguments can be passed to the `train.py` script:

- `--batch_size`: Batch size for training (default: 128)
- `--num_workers`: Number of workers for data loading (default: 2)
- `--learning_rate`: Initial learning rate (default: 5e-4)
- `--warmup_epochs`: Number of warmup epochs for learning rate (default: 10)
- `--epochs`: Total number of training epochs (default: 200)
- `--device`: Device to use for training, either "cpu", "cuda", or "mps" (default: "mps")
- `--image_size`: Size of the input image (default: 32)
- `--patch_size`: Size of the patches to divide the image into (default: 4)
- `--n_classes`: Number of output classes (default: 10)

For a full list of arguments, refer to the [train.py](./train.py) file.

## Model Architecture

The Vision Transformer model implemented in this repository consists of the following key components:

- **Embedding Layer**: Converts image patches into vector embeddings.
- **Transformer Encoder**: Processes embeddings with self-attention and feedforward layers.
- **Classification Head**: A token added to the sequence for final classification.

For details, check the implementation in [model.py](./model.py).

## Results

|          Pre-trained Model          |    Platform     | Test Accuracy | Test Loss |                             Hugging Face Link                              |
| :---------------------------------: | :-------------: | :-----------: | :-------: | :------------------------------------------------------------------------: |
| vit-classification-cifar10-colab-t4 | Google Colab T4 |    78.01%     |  0.6402   | [link](https://huggingface.co/nickpai/vit-classification-cifar10-colab-t4) |
|  vit-classification-cifar10-mbp-m1  | M1 MacBook Pro  |    71.04%     |  0.8440   |  [link](https://huggingface.co/nickpai/vit-classification-cifar10-mbp-m1)  |

```bash
./ViT-Classification-CIFAR10
├── data
├── model
│   ├── vit-classification-cifar10-colab-t4
│   │   └── ViT_model_199.pt
│   └── vit-classification-cifar10-mbp-m1
│       └── ViT_model_199.pt
├── output
│   ├── cifar10-colab-t4
│   │   ├── graph_accuracy.png
│   │   └── graph_loss.png
│   └── cifar10-mbp-m1
│       ├── graph_accuracy.png
│       └── graph_loss.png
├── LICENSE
├── README.md
├── requirements.txt
├── model.py
└── train.py
```

## References

This implementation is inspired by the Vision Transformer paper and other open-source implementations:

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch-Scratch-Vision-Transformer-ViT](https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT)
- [Step-by-Step Guide to Image Classification with Vision Transformers (ViT)](https://comsci.blog/posts/vit)
- [Vision Transformers from Scratch (PyTorch): A step-by-step guide](https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
