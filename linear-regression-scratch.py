# -*- coding: gbk -*-
#%matplotlib inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

def use_svg_display():
    # ��ʸ��ͼ��ʾ
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # ����ͼ�ĳߴ�
    plt.rcParams['figure.figsize'] = figsize


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)

print(features[0], labels[0])

# # ��../d2lzh_pytorch���������������������Ϳ�����������
# import sys
# sys.path.append("..")
# from d2lzh_pytorch import * 

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
plt.scatter(features[:, 0].numpy(), labels.numpy(), 1);
#plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # �����Ķ�ȡ˳���������
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # ���һ�ο��ܲ���һ��batch
        #print(i)
        #print(j)
        yield  features.index_select(0, j), labels.index_select(0, j)
batch_size = 10

#for X, y in data_iter(batch_size, features, labels):
#    print(X, '\n', y)
#    print("end")

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 
#print(w)
#print(b)

def linreg(X, w, b):  # �������ѱ�����d2lzh���з����Ժ�ʹ��
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):  # �������ѱ�����pytorch_d2lzh���з����Ժ�ʹ��
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):  # �������ѱ�����d2lzh_pytorch���з����Ժ�ʹ��
    for param in params:
        param.data -= lr * param.grad / batch_size # ע���������paramʱ�õ�param.data
        
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # ѵ��ģ��һ����Ҫnum_epochs����������
    # ��ÿһ�����������У���ʹ��ѵ�����ݼ�����������һ�Σ������������ܹ���������С��������X
    # ��y�ֱ���С���������������ͱ�ǩ
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l���й�С����X��y����ʧ
        l.backward()  # С��������ʧ��ģ�Ͳ������ݶ�
        sgd([w, b], lr, batch_size)  # ʹ��С��������ݶ��½�����ģ�Ͳ���
        
        # ��Ҫ�����ݶ�����
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f ' % (epoch + 1, train_l.mean().item()))