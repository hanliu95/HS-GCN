import pickle
import torch
import math
import numpy as np
from HSGCN_model import HSGCN
from torch_geometric.data import Data
import torch.utils.data as D
import torch.nn.functional as F

f_para = open('./para/movie_load.para', 'rb')
para_load = pickle.load(f_para)
user_num = para_load['user_num']  # total number of users
item_num = para_load['item_num']  # total number of items
print('total number of users is ', user_num)
print('total number of items is ', item_num)
edge_index = para_load['edge_index']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_index = torch.tensor(edge_index, dtype=torch.long)
num_node_features = 64
x = torch.tensor(np.random.normal(scale=1, size=[user_num + item_num, num_node_features]), dtype=torch.float).to(device)
x.requires_grad = True

class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = HSGCN()
        self.conv2 = HSGCN()
        self.sign = LBSign.apply
    def forward(self, data, beta=1, trainning=True):
        x, edge_index = data.x, data.edge_index
        x = torch.tanh(x)
        if trainning:
            h_0 = x
            h = self.sign(x)
        else:
            h_0 = torch.sign(x)
            h = h_0
        h_1 = self.conv1(h_0, edge_index)
        h_2 = self.conv2(h_1, edge_index)
        h_2 = F.dropout(h_2, p=0.1, training=self.training)
        return h, h_2

model = Net().to(device)
data = Data(x=x, edge_index=edge_index.t().contiguous()).to(device)
learning_rate = 3e-4
optimizer = torch.optim.Adam([{'params': model.parameters()},
                              {'params': x}],
                             lr=learning_rate, weight_decay=1e-7)

batch_size = 1000
step_threshold = 600
alpha = 0.2
lamb1 = 0.1
lamb2 = 0.5
epoch_max = 70
data_block = 3

train_i = torch.empty(0).long()
train_j = torch.empty(0).long()
train_m = torch.empty(0).long()
for b_i in list(range(data_block)):
    triple_para = pickle.load(open('./para/movie_triple_' + str(b_i) + '.para', 'rb'))
    train_i = torch.cat((train_i, torch.tensor(triple_para['train_i'])))  # 1-D tensor of user node ID
    train_j = torch.cat((train_j, torch.tensor(triple_para['train_j'])))  # 1-D tensor of pos item node ID
    train_m = torch.cat((train_m, torch.tensor(triple_para['train_m'])))  # 1-D tensor of neg item node ID

train_dataset = D.TensorDataset(train_i, train_j, train_m)
train_loader = D.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

model.train()
for epoch in range(epoch_max):

    running_loss = 0.0

    for step, (batch_i, batch_j, batch_m) in enumerate(train_loader):
        optimizer.zero_grad()
        H, out = model(data)
        embedding_i = out[batch_i.numpy(), :]
        embedding_j = out[batch_j.numpy(), :]
        embedding_m = out[batch_m.numpy(), :]
        H_i = H[batch_i.numpy(), :]
        H_j = H[batch_j.numpy(), :]
        H_m = H[batch_m.numpy(), :]
        predict_ij = torch.sum(torch.mul(embedding_i, embedding_j), dim=1)  # 1-D
        predict_im = torch.sum(torch.mul(embedding_i, embedding_m), dim=1)
        predict = torch.cat((predict_ij, predict_im))
        target = torch.cat((torch.ones_like(batch_j).float(), torch.zeros_like(batch_m).float())).to(device)
        cross_loss = F.binary_cross_entropy_with_logits(predict, target)
        p_ij = torch.sum(torch.mul(H_i, H_j), dim=1)
        p_im = torch.sum(torch.mul(H_i, H_m), dim=1)
        p_0 = torch.cat((p_ij, p_im))
        rank_loss = torch.mean(F.relu(torch.sigmoid(predict_im) - torch.sigmoid(predict_ij) + alpha))
        cross_loss_0 = F.binary_cross_entropy_with_logits(p_0, target)
        rank_loss_0 = torch.mean(F.relu(torch.sigmoid(p_im) - torch.sigmoid(p_ij) + alpha))
        loss = cross_loss + lamb1 * rank_loss + lamb2 * rank_loss_0
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % step_threshold == (step_threshold-1):
            print('[%d, %5d] loss: %.5f' % (epoch + 1, step + 1, running_loss / step_threshold))
            running_loss = 0.0

model.eval()

with torch.no_grad():
    _, hash_codes = model(data, trainning=False)
    hash_codes = hash_codes.int().cpu().numpy()
    print(hash_codes)
    para = {}
    para['hash_codes'] = hash_codes
    pickle.dump(para, open('./para/movie_codes.para', 'wb'))
    print('HSGCN model training finished...')
