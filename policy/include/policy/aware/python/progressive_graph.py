import networkx as nx
import random
import numpy as np

class NodeEmbeddings(object):
  def __init__(self, cost : int, out_size : int, 
               mem : int, placement=None):
    self.cost : int = cost
    self.out_size : int = out_size
    self.placement : int = placement
    self.curr_bit : int = 0
    self.done_bit : int = 0
    self.start_time : float = None

  def update_placement(self, new_placement):
    self.placement = new_placement

  def set_curr_bit(self):
    self.curr_bit = 1

  def reset_curr_bit(self):
    self.curr_bit = 0

  def reset_done_bit(self):
    self.done_bit = 0

  def update_start_time(self, t):
    self.start_time = t

  def get_embedding(self, n_devs):
    l = [self.cost, self.out_size, self.done_bit, self.curr_bit, self.start_time]
    l = l + self.placement_to_one_hot(self.placement, n_devs)
    return l

  def placement_to_one_hot(self, p, n_devs):
    ret = [0] * n_devs
    if p is not None:
      ret[p] = 1
    return ret

  def normalize_start_time(E, curr_node):
    E[:, 4] -= E[curr_node, 4]
    return E

  def normalize(E, factors):
    # normalize cost, out_size, start_time
    for i in [0, 1, 4]:
      if factors[i] != 0:
        E[:, i] /= factors[i]
    return E

  def inc_done_bit(self):
    self.done_bit += 1

class ProgressiveGraph(object):
  def __init__(self, 
               nx_graph : nx.DiGraph(), 
               n_devs : int, 
               node_traversal_order : str, 
               radial_mp : int,
               seed : int = 42):
    self.seed : int = seed
    random.seed(seed)
    self.nx_graph : nx.DiGraph() = nx_graph
    self.n_devs : int = n_devs

    if node_traversal_order == 'topo':
      self.node_traversal_order = list(nx.topological_sort(self.nx_graph))
    elif node_traversal_order == 'random':
      self.node_traversal_order = list(self.nx_graph.nodes())
      random.shuffle(self.node_traversal_order)
    else:
      raise Exception('Node traversal order not specified correctly')

    idx = {}
    for i, node in enumerate(self.node_traversal_order):
      idx[node] = i
    nx.set_node_attributes(self.nx_graph, idx, 'idx')

    for n in self.nodes():
      assert self.nx_graph.nodes[n]['cost'] is not None 
      assert self.nx_graph.nodes[n]['out_size'] is not None
      assert self.nx_graph.nodes[n]['mem'] is not None
      assert self.nx_graph.nodes[n]['device'] is not None
    
    self.init_node_embeddings()
    self.init_positional_mats()
    self.init_adj_mat()
    self.init_badj_fadj()

  def get_embed_size(self):
    return 5 + self.n_devs

  def get_embeddings(self):
    E = []
    for node_embedding in self.node_embeddings:
      E.append(node_embedding.get_embedding(self.n_devs))

    E = np.array(E, dtype=np.float32)
    E = NodeEmbeddings.normalize_start_time(E, self.curr_node)
    self.embeddings = NodeEmbeddings.normalize(E, np.amax(E, axis=0))

    return self.embeddings    

  def nodes(self):
    return self.node_traversal_order
  
  def get_idx(self, node):
    return self.nx_graph.nodes[node]['idx']

  def neighbors(self, node):
    return self.nx_graph.neighbors(node)
  
  def get_nx_graph(self):
    return self.nx_graph
  
  def n_nodes(self):
    return self.nx_graph.number_of_nodes()

  def get_fadj(self):
    return self.fadj
    
  def get_badj(self):
    return self.badj

  def init_node_embeddings(self):
    E = []
    for n in self.nodes():
      e = NodeEmbeddings(self.nx_graph.nodes[n]['cost'], 
                         self.nx_graph.nodes[n]['out_size'], 
                         self.nx_graph.nodes[n]['mem'])
      E.append(e)
    self.node_embeddings : list(NodeEmbeddings) = E

  def init_positional_mats(self):
    path_mat = nx.floyd_warshall_numpy(self.nx_graph, 
                                       nodelist = self.nodes())
    peer_mat : np.ndarray = np.isinf(path_mat)
    for i in range(len(peer_mat)):
      for j in range(len(peer_mat)):
        if i != j:
          peer_mat[i, j] &= peer_mat[j, i]
        else:
          peer_mat[i, j] = False

    self.peer_mat = peer_mat
    self.progenial_mat = np.logical_not(np.isinf(path_mat))
    np.fill_diagonal(self.progenial_mat, 0)
    self.ancestral_mat = self.progenial_mat.T

  def init_adj_mat(self):
    self.adj_mat = nx.to_numpy_array(self.nx_graph, 
                                      nodelist=self.nodes())
    self.undirected_adj_mat = np.array(self.adj_mat)

    for i in range(len(self.adj_mat)):
      for j in range(len(self.adj_mat)):
        self.undirected_adj_mat[i, j] = max(self.undirected_adj_mat[i, j],
                                            self.undirected_adj_mat[j, i])

  def init_badj_fadj(self):
    self.badj = np.float32(nx.to_numpy_array(self.nx_graph, self.nodes()))
    self.fadj = self.badj.transpose()

  def get_zero_placement(self):
    zero_placement = {}
    for node in self.nodes():
      zero_placement[node] = 0
    return zero_placement

  def get_random_placement(self, seed = None):
    random_placement = {}
    if seed:
      random.seed(seed)
    for node in self.nodes():
      random_placement[node] = random.randint(0, self.n_devs - 1)
    return random_placement

  def get_null_placement(self):
    null_placement = {}
    for node in self.nodes():
      null_placement[node] = None
    return null_placement
  
  def reset_placement(self, pl):
    for i, node in enumerate(self.nodes()):
      self.node_embeddings[i].update_placement(pl[node])
    nx.set_node_attributes(self.nx_graph, pl, 'placement')
  
  def new_episode(self):
    for node_embedding in self.node_embeddings:
      node_embedding.reset_curr_bit()
      node_embedding.reset_done_bit()

  def get_placement(self):
    return nx.get_node_attributes(self.nx_graph, 'placement')

  def set_start_times(self, d):
    for i, n in enumerate(self.nodes()):
      self.node_embeddings[i].update_start_time(d[n])

  def set_curr_node(self, node):
    for node_embedding in self.node_embeddings:
      node_embedding.reset_curr_bit()

    i = self.get_idx(node)
    self.node_embeddings[i].set_curr_bit()
    self.curr_node = i

  def get_ancestral_mask(self, node):
    return self.ancestral_mat[self.get_idx(node), :]

  def get_progenial_mask(self, node):
    return self.progenial_mat[self.get_idx(node), :]

  def get_peer_mask(self, node, start_times, n_peers):
    return self.peer_mat[self.get_idx(node), :]

  def get_self_mask(self, node):
    m = np.zeros((1, len(self.nodes())))
    m[:, self.get_idx(node)] = 1.
    return m

  def inc_done_node(self, node):
    i = self.get_idx(node)
    self.node_embeddings[i].inc_done_bit()

  def refresh(self, nodes, new_p):
    for p, node in zip(new_p, nodes):
      self.nx_graph.nodes[node]['placement'] = p
      i = self.get_idx(node)
      self.node_embeddings[i].update_placement(p)

  def update_start_times(self, start_times):
    for i, node in enumerate(self.nodes()):
      self.node_embeddings[i].update_start_time(start_times[node])