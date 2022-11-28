import torchvision.datasets as datasets
from matplotlib import pyplot as plt
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch import optim
from tqdm import tqdm

mnist = datasets.MNIST(root='~', train=True, download=True)

for i, j in enumerate(np.random.randint(0, len(mnist), (10,))):
    data, label = mnist[j]
    plt.subplot(2, 5, i + 1)
    plt.imshow(data)

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

normalized = trans(mnist[0][0])

mnist = datasets.MNIST(root='~', train=True, download=True, transform=trans)


def imshow(img):
    img = img * 0.3081 + 0.1307
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataloader = DataLoader(mnist, batch_size=4, shuffle=True, num_workers=4)
images, labels = next(iter(dataloader))

imshow(torchvision.utils.make_grid(images))


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.inputlayer = nn.Sequential(nn.Linear(28 * 28, 256), nn.ReLU(), nn.Dropout(0.2))
        self.hiddenlayer = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2))
        self.outlayer = nn.Sequential(nn.Linear(256, 10))

    def forward(self, x):
        # 将输入图像拉伸为一维向量
        x = x.view(x.size(0), -1)

        x = self.inputlayer(x)
        x = self.hiddenlayer(x)
        x = self.outlayer(x)
        return x


# 数据处理和加载
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root='~', train=True, download=True, transform=trans)
mnist_val = datasets.MNIST(root='~', train=False, download=True, transform=trans)

trainloader = DataLoader(mnist_train, batch_size=16, shuffle=True, num_workers=4)
valloader = DataLoader(mnist_val, batch_size=16, shuffle=True, num_workers=4)

# 模型
model = MLP()

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 损失函数
celoss = nn.CrossEntropyLoss()
best_acc = 0


# 计算准确率
def accuracy(pred, target):
    pred_label = torch.argmax(pred, 1)
    correct = sum(pred_label == target).to(torch.float)
    # acc = correct / float(len(pred))
    return correct, len(pred)


acc = {'train': [], "val": []}
loss_all = {'train': [], "val": []}

for epoch in tqdm(range(10)):
    # 设置为验证模式
    model.eval()
    numer_val, denumer_val, loss_tr = 0., 0., 0.
    with torch.no_grad():
        for data, target in valloader:
            output = model(data)
            loss = celoss(output, target)
            loss_tr += loss.data

            num, denum = accuracy(output, target)
            numer_val += num
            denumer_val += denum
    # 设置为训练模式
    model.train()
    numer_tr, denumer_tr, loss_val = 0., 0., 0.
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = celoss(output, target)
        loss_val += loss.data
        loss.backward()
        optimizer.step()
        num, denum = accuracy(output, target)
        numer_tr += num
        denumer_tr += denum
    loss_all['train'].append(loss_tr / len(trainloader))
    loss_all['val'].append(loss_val / len(valloader))
    acc['train'].append(numer_tr / denumer_tr)
    acc['val'].append(numer_val / denumer_val)

'''
>>>   0%|          | 0/10 [00:00<?, ?it/s]
>>>  10%|█         | 1/10 [00:16<02:28, 16.47s/it]
>>>  20%|██        | 2/10 [00:31<02:07, 15.92s/it]
>>>  30%|███       | 3/10 [00:46<01:49, 15.68s/it]
>>>  40%|████      | 4/10 [01:01<01:32, 15.45s/it]
>>>  50%|█████     | 5/10 [01:15<01:15, 15.17s/it]
>>>  60%|██████    | 6/10 [01:30<01:00, 15.19s/it]
>>>  70%|███████   | 7/10 [01:45<00:44, 14.99s/it]
>>>  80%|████████  | 8/10 [01:59<00:29, 14.86s/it]
>>>  90%|█████████ | 9/10 [02:15<00:14, 14.97s/it]
>>> 100%|██████████| 10/10 [02:30<00:00, 14.99s/it]
'''

plt.plot(loss_all['train'])
plt.plot(loss_all['val'])

plt.plot(acc['train'])
plt.plot(acc['val'])

plt.savefig('cnn.png')