import os.path as osp

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.modules.module import Module
import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from torch_sparse import coalesce, SparseTensor, matmul

from opengsl.utils.utils import sparse_tensor_to_scipy_sparse, scipy_sparse_to_sparse_tensor
from opengsl.data.preprocess.normalize import normalize

class BaseAttack(Module):
    """Abstract base class for target attack classes.

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    """

    def __init__(self, model, nnodes, attack_structure=True, attack_features=False, device='cpu'):
        super(BaseAttack, self).__init__()

        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device
        self.modified_adj = None
        self.modified_features = None
        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.nhid

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """Generate attacks on the input graph.

        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of edge removals/additions.

        Returns
        -------
        None.

        """
        pass

    def check_adj(self, adj):
        """Check if the modified adjacency is symmetric and unweighted.
        """
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.tocsr().max() == 1, "Max value should be 1!"
        assert adj.tocsr().min() == 0, "Min value should be 0!"

    def check_adj_tensor(self, adj):
        """Check if the modified adjacency is symmetric, unweighted, all-zero diagonal.
        """
        assert torch.abs(adj - adj.t()).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1, "Max value should be 1!"
        assert adj.min() == 0, "Min value should be 0!"
        diag = adj.diag()
        assert diag.max() == 0, "Diagonal should be 0!"
        assert diag.min() == 0, "Diagonal should be 0!"


    def save_adj(self, root=r'/tmp/', name='mod_adj'):
        """Save attacked adjacency matrix.

        Parameters
        ----------
        root :
            root directory where the variable should be saved
        name : str
            saved file name

        Returns
        -------
        None.

        """
        assert self.modified_adj is not None, \
                'modified_adj is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_adj = self.modified_adj

        if type(modified_adj) is torch.Tensor:
            sparse_adj = sparse_tensor_to_scipy_sparse(modified_adj)
            sp.save_npz(osp.join(root, name), sparse_adj)
        else:
            sp.save_npz(osp.join(root, name), modified_adj)

    def save_features(self, root=r'/tmp/', name='mod_features'):
        """Save attacked node feature matrix.

        Parameters
        ----------
        root :
            root directory where the variable should be saved
        name : str
            saved file name

        Returns
        -------
        None.

        """

        assert self.modified_features is not None, \
                'modified_features is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_features = self.modified_features

        if type(modified_features) is torch.Tensor:
            sparse_features = sparse_tensor_to_scipy_sparse(modified_features)
            sp.save_npz(osp.join(root, name), sparse_features)
        else:
            sp.save_npz(osp.join(root, name), modified_features)

class BaseMeta(BaseAttack):
    """Abstract base class for meta attack. Adversarial Attacks on Graph Neural
    Networks via Meta Learning, ICLR 2019,
    https://openreview.net/pdf?id=Bylnx209YX

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    lambda_ : float
        lambda_ is used to weight the two objectives in Eq. (10) in the paper.
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    undirected : bool
        whether the graph is undirected
    device: str
        'cpu' or 'cuda'
            // TO modify

    """

    def __init__(self, model, with_relu, nnodes=None, feature_shape=None, lambda_=0.5, attack_structure=True, attack_features=False, undirected=True, device='cpu'):

        super(BaseMeta, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.lambda_ = lambda_

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            self.undirected = undirected
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes)).to(device)
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert feature_shape is not None, 'Please give feature_shape='
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape)).to(device)
            self.feature_changes.data.fill_(0)

        self.with_relu = with_relu

    def attack(self, adj, labels, n_perturbations):
        pass

    def get_modified_adj(self, ori_adj):
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        # ind = np.diag_indices(self.adj_changes.shape[0]) # this line seems useless
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        modified_adj = adj_changes_square + ori_adj
        return modified_adj

    def get_modified_features(self, ori_features):
        return ori_features + self.feature_changes

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask

    def self_training_label(self, output, labels, idx_train):
        # Predict the labels of the unlabeled nodes to use them for self-training.
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.

        Note that different data type (float, double) can effect the final results.
        """
        t_d_min = torch.tensor(2.0).to(self.device)
        if self.undirected:
            t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        else:
            t_possible_edges = np.array((np.ones((self.nnodes, self.nnodes)) - np.eye(self.nnodes)).nonzero()).T
        allowed_mask, current_ratio = likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff, undirected=self.undirected)
        return allowed_mask, current_ratio

    def get_adj_score(self, adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_meta_grad -= adj_meta_grad.min()
        # Filter self-loops
        adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
        # # Set entries to 0 that could lead to singleton nodes.
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad *  singleton_mask

        if ll_constraint:
            allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
            allowed_mask = allowed_mask.to(self.device)
            adj_meta_grad = adj_meta_grad * allowed_mask
        return adj_meta_grad

    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad


class Metattack(BaseMeta):
    """Meta attack. Adversarial Attacks on Graph Neural Networks
    via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import Metattack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
    >>> modified_adj = model.modified_adj
            // TO modify

    """

    def __init__(self, model, with_relu,  nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9):

        super(Metattack, self).__init__(model, with_relu, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.hidden_sizes = [model.nhid]
        self.nfeat = model.nfeat
        self.nclass = model.nclass

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
            output_b_velocity = torch.zeros(output_bias.shape).to(device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad

    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):
        """Generate n_perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_unlabeled:
            unlabeled nodes indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        ll_constraint: bool
            whether to exert the likelihood ratio test constraint
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
            is False.
            // TO modify
        """
        output = self.surrogate([ori_features, ori_adj, True])
        self.sparse_features = sp.issparse(ori_features)
        # ori_adj, ori_features, labels = to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(output, labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features
        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = normalize(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)
            
            if self.attack_structure:
                adj_grad = adj_grad.to(self.device)
            if self.attack_features:
                feature_grad = feature_grad.to(self.device)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()


def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004, undirected=True):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.
    """

    N = int(modified_adjacency.shape[0])
    # original_degree_sequence = get_degree_squence(original_adjacency)
    # current_degree_sequence = get_degree_squence(modified_adjacency)
    original_degree_sequence = original_adjacency.sum(0)
    current_degree_sequence = modified_adjacency.sum(0)
    concat_degree_sequence = torch.stack((current_degree_sequence, original_degree_sequence))

    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence, d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(current_degree_sequence, d_min)

    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(concat_degree_sequence, d_min)

    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.
    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               modified_adjacency, d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(n_combined, new_sum_log_degrees_combined, d_min)
    new_ll_combined = compute_log_likelihood(n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold

    if allowed_edges.is_cuda:
        filtered_edges = node_pairs[allowed_edges.cpu().numpy().astype(bool)]
    else:
        filtered_edges = node_pairs[allowed_edges.numpy().astype(bool)]

    allowed_mask = torch.zeros(modified_adjacency.shape)
    allowed_mask[filtered_edges.T] = 1
    if undirected:
        allowed_mask += allowed_mask.clone().t()
    return allowed_mask, current_ratio

def compute_alpha(n, sum_log_degrees, d_min):
    try:
        alpha =  1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha =  1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha

def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees

    return ll

def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    """ Adopted from https://github.com/danielzuegner/nettack
    """
    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]
    degree_sequence = adjacency_matrix.sum(1)
    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)
    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)
    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)
    return new_ll, new_alpha, new_n, sum_log_degrees_after

def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.
    """

    # Determine which degrees are to be considered, i.e. >= d_min.
    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees

def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (torch.log(torch.clamp(d_old_in_range, min=1))).sum(1) \
                                 + (torch.log(torch.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min

    new_n = n_old - (old_in_range!=0).sum(1) + (new_in_range!=0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = scipy_sparse_to_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = scipy_sparse_to_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)


def unravel_index(index, array_shape):
    rows = torch.div(index, array_shape[1], rounding_mode='trunc')
    cols = index % array_shape[1]
    return rows, cols

