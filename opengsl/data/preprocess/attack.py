import torch
import torch.nn.functional as F
from opengsl.utils.utils import accuracy
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from opengsl.data.preprocess.normalize import normalize
from opengsl.utils.utils import scipy_sparse_to_sparse_tensor, sparse_tensor_to_scipy_sparse
import numpy as np
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack, Random
from deeprobust.graph.targeted_attack import Nettack
from tqdm import tqdm



def metattack(adj, feats, labels, nclass, train_mask, val_mask, test_mask, conf, device):
    """
    adj : dense tensor
    
    """
    
    attack_structure = conf.attack['attack_structure']
    attack_features = conf.attack['attack_features']
    lambda_ = conf.attack['lambda_']
    ptb_rate = conf.attack['ptb_rate']
    idx_unlabeled = np.union1d(val_mask, test_mask)
    perturbations = int(ptb_rate * (adj.sum()//2))
    surrogate = GCN(nfeat=feats.shape[1], nclass=nclass, nhid=16,
            dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device).to(device)
    surrogate.fit(feats, adj, labels, train_mask)
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=feats.shape,  attack_structure=attack_structure, attack_features=attack_features, device=device, lambda_=lambda_).to(device)
    model.attack(feats, adj, labels, train_mask, idx_unlabeled, perturbations, ll_constraint=False)
    if attack_structure == True :
        new_adj = model.modified_adj
    else :
        new_adj = adj
    if attack_features == True : 
        new_feats = model.modified_features
    else :
        new_feats = feats
    return new_adj, new_feats

def nettack(adj, feats, labels, nclass, target_node_list, train_mask, val_mask, test_mask, conf, device):
    """
    adj : dense tensor
    
    """

    attack_structure = conf.attack['attack_structure']
    attack_features = conf.attack['attack_features']
    n_perturbations = conf.attack['n_perturbations']
    surrogate = GCN(nfeat=feats.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
    surrogate.fit(feats, adj, labels, train_mask, val_mask, patience=30)
    degrees = adj.sum(0)
    new_adj = adj
    new_feats = feats
    for target_node in tqdm(target_node_list):
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=attack_structure, attack_features=attack_features, device=device).to(device)
        model.attack(new_feats, new_adj, labels, target_node, n_perturbations, verbose=False)
        if attack_structure == True :
            new_adj = scipy_sparse_to_sparse_tensor(model.modified_adj).to_dense().to(device)
        if attack_features == True : 
            new_feats = scipy_sparse_to_sparse_tensor(model.modified_features).to_dense().to(device)
    return new_adj, new_feats

def select_nodes(adj, test_mask, p):
    degrees = adj.sum(0)
    target_node_list = []
    for node in test_mask:
        if (degrees[node] >= 10):
            target_node_list.append(node)
    return target_node_list

def random_attack(adj, conf, device):
    model = Random()

    n_perturbations = int(conf.attack['ptb_rate'] * (adj.sum()//2))
    model.attack(sparse_tensor_to_scipy_sparse(adj), n_perturbations)
    return scipy_sparse_to_sparse_tensor(model.modified_adj).to_dense().to(device)
