import torch
import torchvision
import torchvision.transforms as transforms

# trainset = torchvision.datasets.MNIST(root='.', train=True, download=True)
testset = torchvision.datasets.MNIST(root='.', train=False, download=True)

print(type(testset[0]))
