from AlexNet import AlexNet
from util import MNISTDataSet
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import optim
from torch import nn, Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = MNISTDataSet(root='./data', train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

test_set = MNISTDataSet(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=4)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % str(labels[j]) for j in range(4)))

net = AlexNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epoch = 100

for _ in range(epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss: Tensor = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
