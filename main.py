import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from AlexNet import AlexNet
from util import MNISTDataSet

batch_size = 256
epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = MNISTDataSet(root='./data', train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = MNISTDataSet(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % str(labels[j]) for j in range(4)))

net = AlexNet()
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net.train()
loss_count = []
for s in range(epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss: Tensor = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 300 == 233:
            print('[第%d轮] 平均loss: %.3f' % (s + 1, running_loss / (len(train_set) / batch_size)))
            loss_count.append(running_loss)
            running_loss = 0.0

plt.plot(loss_count)
plt.ylabel('loss')
plt.xlabel('训练次数')
plt.show()

torch.save(net, 'data/alexnet.pth')
# net = torch.load('data/alexnet.pth')
# net.to(device)

net.eval()
correct = 0
total = 0

class_total = list(0 for i in range(10))
class_correct = list(0 for i in range(10))

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            class_total[labels[i].item()] += 1
            if predicted[i] == labels[i]:
                class_correct[labels[i].item()] += 1
    print('Accuracy of the network on the 10000 test images: %.4f%%' % (100.0 * correct / total))

for i in range(10):
    print("Accuracy of %d : %.4f%%" % (i, 1.0 * class_correct[i] / class_total[i]))
