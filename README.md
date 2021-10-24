# IDE-FSNN
This is the PyTorch implementation of paper: Training Feedback Spiking Neural Networks by Implicit Differentiation on the Equilibrium State **(NeurIPS 2021 Spotlight)**. [arxiv](https://arxiv.org/abs/2109.14247).

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch, torchvision](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python scipy termcolor matplotlib progress`

## Dataset Preparation
As for MNIST and Fashion-MNIST, the data can be downloaded by torchvision as in the code. The cifar-10 and cifar-100 datasets are avaliable at [here](https://www.cs.toronto.edu/~kriz/cifar.html). The N-MNIST dataset is avaliable at [here](https://www.garrickorchard.com/datasets/n-mnist).

We use the preprocessing code for N-MNIST from the [TSSL-BP](https://github.com/stonezwr/TSSL-BP/tree/master/preprocessing/NMNIST) repository. We preprocess the data by this code and then place the data in the /data/NMNIST/.

## Training
Run as following with some hyperparameters:

	python mnist_conv.py --gpu-id 0 --path path_to_data --time_step 30 --leaky 0.99 -c checkpoint_name

	python cifar_alexnetf.py --gpu-id 0 --dataset cifar10 --path path_to_data --time_step 30 --leaky 0.99 -c checkpoint_name

As for the IF model, the leaky term could be set as 1, and for the LIF model, the leaky term should be in the range of (0, 1). The default hyperparameters in the code are the same as in the paper.

## Testing
Run as following with some hyperparameters:

	python cifar_alexnetf.py --gpu-id 0 --dataset cifar10 --path path_to_data --time_step 30 --leaky 0.99 -c checkpoint_name --resume path_to_checkpoint --evaluate

We also provide the example code to calculate the firing rate statistics during evaluation. Run as following:

	python get_rate_cifar_alexnetf.py --gpu-id 0 --dataset cifar10 --path path_to_data --time_step 30 --leaky 0.99 --resume path_to_checkpoint

Some pretrained models on CIFAR-10 and CIFAR-100 can be downloaded from [google drive](https://drive.google.com/drive/folders/1ODzSMLSG9yh6plLjgtJyBAanIuDY54EP?usp=sharing).

## Results
The results of our method on MNIST, Fashion-MNIST, N-MNIST, CIFAR-10 and CIFAR-100 are:

### MNIST
| Neuron Model | Network Structure | Time steps | Mean | Std | Best | Neurons | Params |
| :-----------------------: | :-----: | :-----: | :----: | :-----: | :------: | :--------: | :-----: |
| IF | 64C5s (F64C5) | 30 | 99.49% | 0.04% | 99.55% | 13K | 229K |
| LIF (leaky=0.95) | 64C5s (F64C5) | 30 | 99.53% | 0.04% | 99.59% | 13K | 229K |

### Fashion-MNIST
| Neuron Model | Network Structure | Time steps | Mean | Std | Best | Neurons | Params |
| :-----------------------: | :-----: | :-----: | :----: | :-----: | :------: | :--------: | :-----: |
| IF | 400 (F400) | 5 | 90.04% | 0.09% | 90.14% | 1.2K | 478K |
| LIF (leaky=0.95) | 400 (F400) | 5 | 90.07% | 0.10% | 90.25% | 1.2K | 478K |

### N-MNIST
| Neuron Model | Network Structure | Time steps | Mean | Std | Best | Neurons | Params |
| :-----------------------: | :-----: | :-----: | :----: | :-----: | :------: | :--------: | :-----: |
| IF | 64C5s (F64C5) | 30 | 99.30% | 0.04% | 99.35% | 21K | 291K |
| LIF (leaky=0.95) | 64C5s (F64C5) | 30 | 99.42% | 0.04% | 99.47% | 21K | 291K |

### CIFAR-10
| Neuron Model | Network Structure | Time steps | Mean | Std | Best | Neurons | Params |
| :-----------------------: | :-----: | :-----: | :----: | :-----: | :------: | :--------: | :-----: |
| IF | AlexNet-F | 30 | 91.73% | 0.13% | 91.85% | 159K | 3.7M |
| LIF (leaky=0.99) | AlexNet-F | 30 | 91.74% | 0.09% | 91.92% | 159K | 3.7M |
| IF | AlexNet-F | 100 | 92.25% | 0.27% | 92.53% | 159K | 3.7M |
| LIF (leaky=0.99) | AlexNet-F | 100 | 92.03% | 0.07% | 92.15% | 159K | 3.7M |
| IF | CIFARNet-F | 30 | 91.94% | 0.14% | 92.12% | 232K | 11.8M |
| LIF (leaky=0.99) | CIFARNet-F | 30 | 92.08% | 0.15% | 92.23% | 232K | 11.8M |
| IF | CIFARNet-F | 100 | 92.33% | 0.15% | 92.57% | 232K | 11.8M |
| LIF (leaky=0.99) | CIFARNet-F | 100 | 92.52% | 0.17% | 92.82% | 232K | 11.8M |

### CIFAR-100
| Neuron Model | Network Structure | Time steps | Mean | Std | Best | Neurons | Params |
| :-----------------------: | :-----: | :-----: | :----: | :-----: | :------: | :--------: | :-----: |
| IF | AlexNet-F | 100 | 72.02% | 0.16% | 72.23% | 159K | 5.2M |
| IF | CIFARNet-F | 30 | 71.56% | 0.31% | 72.10% | 232K | 14.8M |
| LIF (leaky=0.99) | CIFARNet-F | 30 | 71.72% | 0.22% | 72.03% | 232K | 14.8M |
| IF | CIFARNet-F | 100 | 73.07% | 0.21% | 73.43% | 232K | 14.8M |
| LIF (leaky=0.99) | CIFARNet-F | 100 | 72.98% | 0.13% | 73.12% | 232K | 14.8M |

## Acknowledgement

The codes for the broyden's method and some utils are modified from the [DEQ](https://github.com/locuslab/deq) and [MDEQ](https://github.com/locuslab/mdeq) repositories. The codes for some utils are from the [pytorch-classification](https://github.com/bearpaw/pytorch-classification) repository.

## Contact
If you have any questions, please contact <mingqing_xiao@pku.edu.cn>.
