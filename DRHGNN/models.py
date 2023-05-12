from tqdm import tqdm
import torch
from itertools import chain
import torch 
import torch.nn as nn 
import math
import torch.nn.functional as F


class HGNNConv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        x = x.mm(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = A.mm(x)
        return x

class HGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, dropout=0.5):
        super().__init__()
        self.hgc1 = HGNNConv(nfeat, nhid)
        self.hgc2 = HGNNConv(nhid, nclass)
        self.convs = torch.nn.ModuleList(
            [HGNNConv(nfeat, nhid)] + 
            [HGNNConv(nhid, nhid) for _ in range(nlayer-2)] +
            [HGNNConv(nhid, nclass)]
        )
        self.args = args 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H):

        for _, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, H))
            x = self.dropout(x)
        x = self.convs[-1](x, H)
        return F.log_softmax(x, dim=1)



class ResHGNNConv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.W = nn.Linear(in_ft, out_ft, bias=bias)

    def forward(self, X, A, alpha, beta, X0):

        X = A.mm(X)  # 将X0修改为X (2023.3.17)
        Xi = (1 - alpha) * X + alpha * X0
        X = (1 - beta) * Xi + beta * self.W(Xi)

        # ablation study
        # HGNN with initial residual
        # X = A.mm(X)
        # X = self.W((1 - alpha) * X + alpha * X0)
        # HGNN with identity mapping
        # X = A.mm(X)
        # X = (1 - beta) * X + beta * self.W(X)

        return X


class ResHGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, dropout=0.5):

        super().__init__()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(nlayer):
            self.convs.append(ResHGNNConv(nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, nclass))



    def forward(self, x, G):
        # lamda, alpha = 0.5, 0.1
        lamda, alpha = 2.5, 0.1
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x
        for i,con in enumerate(self.convs[1:-1]):
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, G, alpha, beta, x0))
        x = self.dropout(x)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)









def general_outcome_correlation(A, y, alpha, num_propagations, post_step, alpha_term, display=False):
    """general outcome correlation. 
    alpha_term = True for outcome correlation, 
    alpha_term = False for residual correlation
    """
    # y: [N, C],  A: [N, N],
    # W_hat = alpha * S * W_hat + W_0
    # W_hat = alpha * S * W_hat + (1 - alpha) * W_0

    result = y.clone()
    addterm = (1-alpha) * y if alpha_term else y
    for _ in tqdm(range(num_propagations), disable=not display):
        result = alpha * (A @ result) + addterm
        result = post_step(result)
    return result


def label_propagation(A, Ytr, alpha, num_propagations=50):
    return general_outcome_correlation(A, Ytr, alpha, num_propagations, post_step=lambda x: torch.clamp(x, 0, 1), alpha_term=True)



import torch.nn as nn
class iHL(nn.Module):
    def __init__(self, nfeat, nclass):
        super().__init__()
        self.W = nn.Linear(nfeat, nclass)
    def forward(self, X):
        X = self.W(X)
        return X 
