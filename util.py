import struct

import torch
from PIL import Image
from torchvision.datasets import VisionDataset

train_image_p = 'data/train-images.idx3-ubyte'
train_label_p = 'data/train-labels.idx1-ubyte'
t10k_image_p = 'data/t10k-images.idx3-ubyte'
t10k_label_p = 'data/t10k-labels.idx1-ubyte'


class MNISTDataSet(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(MNISTDataSet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.image = None
        self.label = None
        if train:
            self.image = load_image(train_image_p)
            self.label = load_label(train_label_p)
        else:
            self.image = load_image(t10k_image_p)
            self.label = load_label(t10k_label_p)

    def __getitem__(self, item):
        img, lab = self.image[item], self.label[item]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lab = self.target_transform(lab)
        return img, lab

    def __len__(self):
        return len(self.label)


def load_image(filename):
    with open(filename, mode='rb') as image_f:
        buf = image_f.read()
    offset = 0
    magic, image_num, width, height = struct.unpack_from('>4I', buf, offset)
    offset += struct.calcsize('>4I')
    # images = np.zeros((image_num, width, height), dtype=np.uint8)
    # 刚开始读入主存中
    images = torch.zeros((image_num, width, height), dtype=torch.uint8)
    fmt = '>' + str(width * height) + 'B'
    for i in range(image_num):
        images[i] = torch.tensor(struct.unpack_from(fmt, buf, offset), dtype=torch.uint8).reshape(
            (width, height))
        offset += struct.calcsize(fmt)
    return images


def load_label(filename):
    with open(filename, mode='rb') as label_f:
        buf = label_f.read()
    offset = 0
    magic, label_num = struct.unpack_from('>2I', buf, offset)
    offset += struct.calcsize('2I')
    # label不需要用Tensor装载
    labels = []
    for i in range(label_num):
        labels.append(struct.unpack_from('>B', buf, offset)[0])
        offset += struct.calcsize('>B')
    return labels
