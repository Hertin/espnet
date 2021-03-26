import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import logging
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import networkx as nx
import numpy as np
from nodevectors import Node2Vec
import scipy.sparse as sp

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def get_normalized_adj(g):
    adj = nx.adjacency_matrix(g)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx, dtype='float'):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    if dtype == 'float':
      return torch.sparse.FloatTensor(indices, values, shape)
    elif dtype == 'half':
      logging.warning('convert to half')
      return torch.sparse.FloatTensor(indices, values, shape).half()
    else:
      raise ValueError('Data type not implemented')

class GraphConvolutionLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # logging.warning(f'adj types {type(adj)}  {adj.type()} support {type(support)} {support.type()}')
        output = torch.spmm(adj, support.float()) #  spmm  does not support half
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class LangGCN(nn.Module):
  """ Graph Convolutional Networks,
      see reference below for more information

      Kipf, T.N. and Welling, M., 2016.
      Semi-supervised classification with graph convolutional networks.
      arXiv preprint arXiv:1609.02907.
  """
  

  def __init__(self, args):
    super(LangGCN, self).__init__()
    self.args = args
    if args.use_glottoph:
      with open(args.glotto_all_phonemes, 'r') as f:
        self.glotto_all_phonemes = sorted(json.load(f))
      self.input_dim = args.lgcn_n2v_dim + len(self.glotto_all_phonemes)
    else:
      self.input_dim = args.lgcn_n2v_dim + len(args.char_list)
    self.hidden_dim = args.lgcn_hidden_dim
    self.output_dim = args.lgcn_output_dim
    self.num_layer = args.lgcn_num_layer
    self.dropout = args.dropout_rate

    dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
    self.gcn_layers = nn.ModuleList([
      GraphConvolutionLayer(
        in_features=dim_list[tt], out_features=dim_list[tt+1]
      ) for tt in range(self.num_layer)
    ])

    self.final_linear = nn.Linear(dim_list[-2], dim_list[-1])

    self.g = nx.read_gpickle(args.lgcn_graph_path)
    adj = get_normalized_adj(self.g)
    dtype = 'half' if args.train_dtype in ("O0", "O1", "O2", "O3") else 'float'

    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    self.adj = sparse_mx_to_torch_sparse_tensor(adj).to(device) # dirty hack

    self.args.char_list.index
    self.lid2vid = {str(n): i for i, n in enumerate(self.g.nodes)}

    with open(args.lgcn_lang2lid_path, 'r') as f:
      self.lang2lid = json.load(f)

    # language initial embedding
    # 1. node2vec embedding
    if os.path.exists(f'{args.lgcn_g2v_path}.npy'):
      logging.warning('load g2v npy directly')
      self.n2v_embedding = np.load(f'{args.lgcn_g2v_path}.npy')
    else:
      self.g2v = Node2Vec.load(args.lgcn_g2v_path)
      self.n2v_embedding = np.array([self.g2v.predict(l) for l in self.g.nodes])
    
    # 2. one hot for phoneme used
    if args.use_glottoph:
      with open(args.lang2glottoph, 'r') as f:
        self.lang2ph = json.load(f)
      with open(args.glotto2ph, 'r') as f:
        self.glotto2ph = json.load(f)
      # use glottophoneme
      n_langs = 0
      self.ph_embedding = np.ones((self.g.number_of_nodes(), len(self.glotto_all_phonemes)))
      for lid in self.g.nodes:
        if lid in self.glotto2ph:
          n_langs += 1
          phones = self.glotto2ph[lid]
          for ph in self.glotto_all_phonemes:
            if ph not in phones:
              self.ph_embedding[self.lid2vid[lid], self.glotto_all_phonemes.index(ph)] = 0
        # otherwise, assume using all phonemes
      logging.warning(f'{n_langs} out of {self.g.number_of_nodes()} nodes has phoneme')
    else:
      # use phonetisaurus phoneme
      self.ph_embedding = np.ones((self.g.number_of_nodes(), len(args.char_list)))
      with open(args.lang2ph, 'r') as f:
        self.lang2ph = json.load(f)
      for lang, phones in self.lang2ph.items():
        phone_set = set(phones + ['<blank>', '<unk>', '<space>', '<eos>'])
        for ph in args.char_list:
          if ph not in phone_set:
            self.ph_embedding[self.lang2vid(lang), args.char_list.index(ph)] = 0
    # 3. concate embeddings
    embedding = np.concatenate([self.n2v_embedding, self.ph_embedding], axis=1)
    self.embedding = Parameter(torch.from_numpy(embedding).float(), requires_grad=False)
    # self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding).float()) # N x D
    # self.embedding.requires_grad_(requires_grad=False)
    # logging.warning(f'{self.embedding.weight[self.lang2vid("PL")]}')


  def lang2vid(self, lang):
    return self.lid2vid[self.lang2lid[lang]]

  def forward(self, langs):
    """      
      Args:
        lang: list of language labels, length B
    """

    state = self.embedding # N x D
    for tt in range(self.num_layer):
      state = self.gcn_layers[tt](state, self.adj)
      state = F.relu(state)
      state = F.dropout(state, self.dropout, training=self.training)
    state = self.final_linear(state) # N x F

    # logging.warning(f'lgcn state size {state.size()}')

    idx = [self.lang2vid(lang) for lang in langs]

    return state[idx]