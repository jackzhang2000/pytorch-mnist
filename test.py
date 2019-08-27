import torch
import torchvision

# trainset = torchvision.datasets.MNIST(root='.', train=True, download=True)
testset = torchvision.datasets.MNIST(root='.', train=False, download=True)

net = torch.load('alexnet.pth')
torch.save(net.state_dict(), 'param.pth')
