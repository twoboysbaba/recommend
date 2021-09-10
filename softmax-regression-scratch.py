# -*- coding: gbk -*-
import torchvision
import matplotlib.pyplot as plt
import sys
import torch
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
import numpy as np

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=torchvision.transforms.ToTensor())
print(len(mnist_train), len(mnist_test))
feature,label = mnist_train[0]
print(feature.shape,feature.dtype)
#torch.Size([1, 28, 28]) torch.float32
#1,获取数据
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#2，初始化模型参数
num_inputs = 784#28*28
num_outputs = 10
W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)), dtype=torch.float32)
b = torch.zeros(num_outputs,dtype=torch.float32)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

#3,定义模型
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition
def net(X,W,b):
    return softmax(torch.mm(X.view(-1,num_inputs),W) + b)

#4,定义损失函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat.gather(1,y.view(-1,1)))
    
def accuracy(y_hat,y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()  

def evaluate_accuracy(data_iter,net):
    acc_sum,n = 0.0,0
    for X,y in data_iter:
        acc_sum = (net(X,W,b).argmax(dim=1) ==y).float().sum().item()
        n +=y.shape[0]
    return acc_sum/n

#5,定义优化函数
def sgd(params,lr,batch_size):
  for param in params:
      param.data -= lr*param.grad/batch_size  
#6,开始训练模型
num_epochs, lr = 5, 0.1

def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum ,train_acc_sum,n = 0.0,0.0,0
        for X,y in train_iter:
            y_hat = net(X,W,b)
            l = loss(y_hat,y).sum()
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                sgd(params,lr,batch_size)
            else:
                optimizer.step()
            train_l_sum +=l.item()
            train_acc_sum +=accuracy(y_hat, y)
            n +=y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d,loss %.4f, train acc %.3f, test acc %.3f'%(epoch+1,train_l_sum/n, train_acc_sum/n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W,b], lr)
'''
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
    
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))
'''