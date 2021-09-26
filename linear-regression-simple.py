# -*- coding: gbk -*-
import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10
# ��ѵ�����ݵ������ͱ�ǩ���
dataset = Data.TensorDataset(features, labels)
# �� dataset ���� DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # Ҫ��Ҫ�������� (���ұȽϺ�)
    num_workers=0,              # ���߳���������
)
for X, y in data_iter:
    print(X, '\n', y)
    break

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net) # ʹ��print���Դ�ӡ������Ľṹ

from torch.nn import init

init.normal_(net.linear.weight, mean=0.0, std=0.01)
init.constant_(net.linear.bias, val=0.0)  # Ҳ����ֱ���޸�bias��data: net[0].bias.data.fill_(0)

loss = nn.MSELoss()

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # �ݶ����㣬�ȼ���net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))