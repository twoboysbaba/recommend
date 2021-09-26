from utils.utils import create_dataset, Trainer
from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from DIEN import DeepInterestEvolutionNetwork
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Training on [{}].'.format(device))

sequence_length = 40
dataset = create_dataset('amazon-books', sample_num=100000, sequence_length=sequence_length, device=device)
field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = dataset.train_valid_test_split()

import numpy as np
from tqdm import tqdm

def auxiliary_sample(X, sample_set):
    pos_sample = X[:, 1: -1]
    neg_sample = torch.zeros_like(pos_sample)
    for i in tqdm(range(pos_sample.shape[0])):
        for j in range(pos_sample.shape[1]):
            if pos_sample[i, j] > 0:
                idx = np.random.randint(len(sample_set))
                while sample_set[idx] == pos_sample[i, j]:
                    idx = np.random.randint(len(sample_set))
                neg_sample[i, j] = sample_set[idx]
            else:
                break
    return  neg_sample


neg_sample = auxiliary_sample(train_X, dataset.cate_set)
train_X_neg = torch.hstack([train_X, neg_sample])

from utils.utils import BatchLoader, EarlyStopper
import matplotlib.pyplot as plt

class DIENTrainer(Trainer):
    
    def __init__(self, model, optimizer, criterion, batch_size=None):
        super(DIENTrainer, self).__init__(model, optimizer, criterion, batch_size)
    
    def train(self, train_X_neg, train_y, epoch=100, trials=None, valid_X=None, valid_y=None):
        if self.batch_size:
            train_loader = BatchLoader(train_X_neg, train_y, self.batch_size)
        else:
            # 为了在 for b_x, b_y in train_loader 的时候统一
            train_loader = [[train_X_neg, train_y]]

        if trials:
            early_stopper = EarlyStopper(self.model, trials)

        train_loss_list = []
        valid_loss_list = []

        for e in tqdm(range(epoch)):
            # train part
            self.model.train()
            train_loss_ = 0
            for b_x, b_y in train_loader:
                self.optimizer.zero_grad()
                seq_len = b_x.shape[1] // 2
                pred_y, auxiliary_y = self.model(b_x[:, :seq_len+1], b_x[:, -seq_len+1:])
                
                auxiliary_true = torch.cat([torch.ones_like(auxiliary_y[0]), torch.zeros_like(auxiliary_y[1])], dim=0).view(2, -1)
                auxiliary_loss = self.criterion(auxiliary_y, auxiliary_true)
                auxiliary_loss.backward(retain_graph=True)
                
                train_loss = self.criterion(pred_y, b_y)
                train_loss.backward()
                
                self.optimizer.step()

                train_loss_ += train_loss.detach() * len(b_x)

            train_loss_list.append(train_loss_ / len(train_X_neg))

            # valid part
            if trials:
                valid_loss, valid_metric = self.test(valid_X, valid_y)
                valid_loss_list.append(valid_loss)
                if not early_stopper.is_continuable(valid_metric):
                    break

        if trials:
            self.model.load_state_dict(early_stopper.best_state)
            plt.plot(valid_loss_list, label='valid_loss')

        plt.plot(train_loss_list, label='train_loss')
        plt.legend()
        plt.show()

        print('train_loss: {:.5f} | train_metric: {:.5f}'.format(*self.test(train_X, train_y)))

        if trials:
            print('valid_loss: {:.5f} | valid_metric: {:.5f}'.format(*self.test(valid_X, valid_y)))
            
EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 600
TRIAL = 100

dien = DeepInterestEvolutionNetwork(field_dims, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(dien.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
criterion = nn.BCELoss()

trainer = DIENTrainer(dien, optimizer, criterion, BATCH_SIZE)
trainer.train(train_X_neg, train_y, epoch=EPOCH, trials=TRIAL, valid_X=valid_X, valid_y=valid_y)
test_loss, test_metric = trainer.test(test_X, test_y)
print('test_loss:  {:.5f} | test_auc:  {:.5f}'.format(test_loss, test_metric))