from opengsl.method.models.metattack import Metattack
from opengsl.method.models.nettack import Nettack
from opengsl.method.models.gcn import GCN
import torch
import torch.nn.functional as F
from opengsl.utils.utils import accuracy
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from opengsl.data.preprocess.normalize import normalize
from opengsl.utils.utils import scipy_sparse_to_sparse_tensor
import numpy as np

def train_surrogate(nhid, nclass, nlayers, dropout, epochs, lr, weight_decay, feats, adj, labels, train_mask, val_mask,  device):
    loss_fn = F.binary_cross_entropy_with_logits if nclass == 1 else F.cross_entropy
    metric = roc_auc_score if nclass == 1 else accuracy
    surrogate = GCN(feats.shape[1], nhid, nclass, nlayers, dropout).to(device)
    surrogate_optim = torch.optim.Adam(surrogate.parameters(), lr=lr,weight_decay=weight_decay)
    loss_fn = F.binary_cross_entropy_with_logits if nclass == 1 else F.cross_entropy
    metric = roc_auc_score if nclass == 1 else accuracy
    best_val = 0
    for epoch in range(epochs):
        surrogate.train()
        surrogate_optim.zero_grad()

        # forward and backward
        output = surrogate([feats, adj, True])
        loss_train = loss_fn(output[train_mask], labels[train_mask])
        acc_train = metric(labels[train_mask].cpu().numpy(),
                                output[train_mask].detach().cpu().numpy())
        acc_val = metric(labels[val_mask].cpu().numpy(), output[val_mask].detach().cpu().numpy())
        if acc_val > best_val:
            best_val = acc_val
            weights = deepcopy(surrogate.state_dict())
        loss_train.backward()
        surrogate_optim.step()
    surrogate.load_state_dict(weights)
    return surrogate


def metattack(adj, feats, labels, nclass, train_mask, val_mask, test_mask, conf, device):
    attack_structure = conf.attack['attack_structure']
    attack_features = conf.attack['attack_features']
    undirected = conf.attack['undirected']
    with_bias = conf.attack['with_bias']
    lambda_ = conf.attack['lambda_']
    train_iters = conf.attack['train_iters']
    lr = conf.attack['lr']
    momentum = conf.attack['momentum']
    if conf.dataset['normalize']:
        gcn_adj = normalize(adj.to_sparse(), add_loop=conf.dataset['add_loop'], sparse=conf.dataset['sparse']).to_dense()
    else:
        gcn_adj = adj
    surrogate = train_surrogate(conf.attack['gcn_nhid'], nclass, conf.attack['gcn_layers'], conf.attack['gcn_dropout'], conf.attack['gcn_epochs'], 
                                conf.attack['gcn_lr'], conf.attack['gcn_weight_decay'], feats, gcn_adj, labels, train_mask,val_mask, device)

    
    model = Metattack(model=surrogate, with_relu = True, nnodes = adj.shape[0], feature_shape = feats.shape,  
                      attack_structure = attack_structure, attack_features = attack_features, undirected = undirected, device = device, lambda_ = lambda_, train_iters = train_iters,
                      with_bias = with_bias, lr = lr, momentum = momentum)
    model.attack(feats, adj, labels, train_mask, np.union1d(val_mask,test_mask), conf.attack['n_perturbations'], conf.attack['ll_constraint'], conf.attack['ll_cutoff'])
    
    # new_adj = adj
    # new_feats = feats
    if attack_structure == True :
        new_adj = model.modified_adj
    else :
        new_adj = adj
    if attack_features == True : 
        new_feats = model.modified_features
    else :
        new_feats = feats
    # print(new_feats)
    # print(feats)
    return new_adj, new_feats

def nettack(adj, feats, labels, nclass, target_node, train_mask, conf, device):

    attack_structure = conf.attack['attack_structure']
    attack_features = conf.attack['attack_features']
    directed = conf.attack['directed']
    surrogate = train_surrogate(conf.attack['gcn_nhid'], nclass, 2, conf.attack['gcn_dropout'], conf.attack['gcn_epochs'], 
                                conf.attack['gcn_lr'], conf.attack['gcn_weight_decay'], feats, adj, labels, train_mask, device)

    model = Nettack(model=surrogate, attack_structure = attack_structure, attack_features = attack_features, device = device)
    model.attack(feats, adj, labels, target_node, conf.attack['n_perturbations'], directed, conf.attack['n_influencers'], conf.attack['ll_cutoff'], conf.attack['verbose'])

    new_adj = adj
    new_feats = feats
    if attack_structure == True :
        new_adj = scipy_sparse_to_sparse_tensor(model.modified_adj).to(device)
    else :
        new_adj = adj
    if attack_features == True : 
        new_feats = model.modified_features
    else :
        new_feats = feats
    return new_adj, new_feats
