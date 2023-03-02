import networkx as nx
import os
import sys
import json
import time
import pickle
from python.progressive_nn import *
from python.progressive_placer import *
from python.simulator import *
import copy
# from progressive_placer import *

class Graph:
  def __init__(self):
    self.graph = nx.DiGraph()
  
  def add_node(self, name, cost, memory, out_size, device):
    # self.graph.add_node(name, cost=cost, mem=memory, 
    #                     out_size=out_size, device=device)
    self.graph.add_node(name)
    self.graph.nodes[name]['cost'] = cost
    self.graph.nodes[name]['mem'] = memory
    self.graph.nodes[name]['out_size'] = out_size
    self.graph.nodes[name]['device'] = device
          
  def add_edge(self, name_from, name_to):
    self.graph.add_edge(name_from, name_to)
  
  def number_of_nodes(self):
    return self.graph.number_of_nodes()
  
  def print_node(self, name):
    return self.graph.nodes[name]

  def get_node_cost(self, name):
    return self.graph.nodes[name]['cost']
  
  def get_node_mem(self, name):
    return self.graph.nodes[name]['mem']

  def get_node_out_size(self, name):
    return self.graph.nodes[name]['out_size']

  def get_node_device(self, name):
    return self.graph.nodes[name]['device']
  
  def get_all_nodes(self):
    return copy.deepcopy(self.graph.nodes)

  def get_nx_graph(self):
    return self.graph

class Pickle:
  def __init__(self, input_path : str):
    self.input_path = input_path
    
    input_file = open(self.input_path, 'rb')
    all_input = pickle.load(input_file)
    self.meta_graph = all_input['optim_mg']
    self.ungrouped_mapping = all_input['ungrouped_mapping']
    self.op_pref = all_input['op_perf']
    self.step_stats = all_input['step_stats']
    if 'G' in all_input:
      self.nx_graph = all_input['G']
    
  def get_meta_graph(self):
    return self.meta_graph
  
  def get_nx_graph(self):
    return self.nx_graph

  def get_ungrouped_mapping(self):
    return self.ungrouped_mapping
  
  def get_op_pref(self):
    return self.op_pref

  def get_step_stats(self):
    return self.step_stats

  def set_nx_graph(self, nx_graph):
    self.nx_graph = nx_graph

class Args:
  def __init__(self, args : dict, reinforce_params : dict):
    self.args = args
    self.seed : int = args.get('seed', None)
    self.name : str = args.get('name', None)
    # self.graph : str = args.get('graph', None)
    self.id : int = args.get('id', None)
    self.graph_size : int = args.get('graph_size', None)
    self.pickled_inp_file : str = args.get('pickled_inp_file', None)
    self.mul_graphs : str = args.get('mul_graphs', None)
    self.dataset_folder : str = args.get('dataset_folder', None)
    self.dataset : str = args.get('dataset', None)
    self.n_devs : int = args.get('n_devs', None)
    self.model_folder_prefix : str = args.get('model_folder_prefix', None)
    self.m_name : str = args.get('m_name', None)
    self.n_peers : int = args.get('n_peers', None)
    # self.agg_msgs : bool = args.get('agg_msgs', None)
    self.no_msg_passing : bool = args.get('no_msg_passing', None)
    self.radial_mp : int = args.get('radial_mp', None)
    # self.tri_agg : bool = args.get('tri_agg', None)
    # self.sage : bool = args.get('sage', None)
    self.sage_hops : int = args.get('sage_hops', None)
    self.sage_sample_ratio : float = args.get('sage_sample_ratio', None)
    self.sage_dropout_rate : float = args.get('sage_dropout_rate', None)
    self.sage_aggregation : str = args.get('sage_aggregation', None)
    self.sage_position_aware : bool = args.get('sage_position_aware', None)
    # self.use_single_layer_perceptron : bool = args.get('use_single_layer_perceptron', None)
    self.pgnn_c : float = args.get('pgnn_c', None)
    self.pgnn_neigh_cutoff : int = args.get('pgnn_neigh_cutoff', None)
    self.pgnn_anchor_exponent : int = args.get('pgnn_anchor_exponent', None)
    self.pgnn_aggregation : str = args.get('pgnn_aggregation', None)
    # self.reinit_model : bool = args.get('reinit_model', None)
    self.n_eps : int = args.get('n_eps', None)
    self.max_rnds : int = args.get('max_rnds', None)
    self.disc_factor : float = args.get('disc_factor', None)
    self.vary_init_state : bool = args.get('vary_init_state', None)
    self.zero_placement_init : bool = args.get('zero_placement_init', None)
    self.null_placement_init : bool = args.get('null_placement_init', None)
    self.init_best_pl : bool = args.get('init_best_pl', None)
    self.one_shot_episodic_rew : bool = args.get('one_shot_episodic_rew', None)
    self.ep_decay_start : float = args.get('ep_decay_start', None)
    self.bl_n_rnds : int = args.get('bl_n_rnds', None)
    self.rew_singlegpu : bool = args.get('rew_singlegpu', None)
    self.rew_neigh_pl : bool = args.get('rew_neigh_pl', None)
    self.supervised : bool = args.get('supervised', None)
    self.use_min_runtime : bool = args.get('use_min_runtime', None)
    self.discard_last_rnds : bool = args.get('discard_last_rnds', None)
    self.turn_based_baseline : bool = args.get('turn_based_baseline', None)
    self.dont_repeat_ff : bool = args.get('dont_repeat_ff', None)
    self.small_nn : bool = args.get('small_nn', None)
    self.dont_restore_softmax : bool = args.get('dont_restore_softmax', None)
    self.restore_from : str = args.get('restore_from', None)
    self.print_freq : int = args.get('print_freq', None)
    self.save_freq : int = args.get('save_freq', None)
    self.eval_freq : int = args.get('eval_freq', None)
    self.log_tb_workers : bool = args.get('log_tb_workers', None)
    self.debug : bool = args.get('debug', None)
    self.debug_verbose : bool = args.get('debug_verbose', None)
    self.disamb_pl : bool = args.get('disamb_pl', None)
    # self.eval : bool = args.get('eval', None)
    # self.simplify_tf_rew_model : bool = args.get('simplify_tf_rew_model', None)
    self.log_runtime : bool = args.get('log_runtime', None)
    # self.use_new_sim : bool = args.get('use_new_sim', None)
    self.gen_profile_timeline : bool = args.get('gen_profile_timeline', None)
    self.mem_penalty : float = args.get('mem_penalty', None)
    self.max_mem : float = args.get('max_mem', None)
    self.max_runtime_mem_penalized : float = args.get('max_runtime_mem_penalized', None)
    self.use_threads : bool = args.get('use_threads', None)
    self.scale_norm : bool = args.get('scale_norm', None)
    self.dont_share_classifier : bool = args.get('dont_share_classifier', None)
    self.use_gpus : bool = args.get('use_gpus', None)
    self.eval_on_transfer : int = args.get('eval_on_transfer', None)
    self.normalize_aggs : bool = args.get('normalize_aggs', None)
    self.bn_pre_classifier : bool = args.get('bn_pre_classifier', None)
    self.bs : int = args.get('bs', None)
    self.num_children : int = args.get('num_children', None)
    self.disable_profiling : bool = args.get('disable_profiling', None)
    self.n_async_sims : int = args.get('n_async_sims', None)
    self.baseline_mask : int = args.get('baseline_mask', None)
    self.n_workers : int = args.get('n_workers', None)
    self.node_traversal_order : str = args.get('node_traversal_order', None)
    # self.prune_final_size : int = args.get('prune_final_size', None)
    self.dont_sim_mem : bool = args.get('dont_sim_mem', None)
    self.remote_async_addrs : str = args.get('remote_async_addrs', None)
    self.remote_async_start_ports : int = args.get('remote_async_start_ports', None)
    self.remote_async_n_sims : int = args.get('remote_async_n_sims', None)
    # self.local_prefix : str = args.get('local_prefix', None)
    self.remote_prefix : str = args.get('remote_prefix', None)
    self.shuffle_gpu_order : bool = args.get('shuffle_gpu_order', None)
    self.tb_dir : str = '%smodels/tb-logs/%s' % (self.model_folder_prefix, self.dataset)
    self.eval_dir : str= '%smodels/eval-dir/%s' % (self.model_folder_prefix, self.dataset)
    self.fig_dir : str = '%s/figs/' % self.eval_dir
    self.record_best_pl_file : str = '%s/best_pl' % self.eval_dir
    self.profiling_chrome_trace : str = '%smodels/chrome-traces/%s/' % (self.model_folder_prefix, self.name)
    self.m_save_path : str = '%smodels/tf-models/%s' % (self.model_folder_prefix, self.name)

    self.reinforce_params : dict = reinforce_params

    self.comfirm_params()
    self.set_env_variable()

  def get_param_value(self, param):
    return getattr(self, param)

  def get_all_params(self):
    return self.__dict__

  def comfirm_params(self):
    if self.one_shot_episodic_rew and self.n_async_sims is not None:
      raise Exception('Input setting leads to deadlock')
    if self.eval_freq % 10 == 0:
      raise Exception('Eval freq cannot be divisible by 10')
  
  def set_env_variable(self):
    if self.use_gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ' '.join(self.use_gpus)
  
  def log_config(self):
    if len(self.name) > 0:
      for d in [self.tb_dir]:
        if os.path.exists(d):
          os.rmdir(d)
    
    for d in [self.tb_dir, self.fig_dir, self.eval_dir, self.profiling_chrome_trace]:
      if not os.path.exists(d):
        os.makedirs(d)

    config = self.get_all_params()
    jsonable_config = {}
    for k, v in config.items():
      if type(v).__name__ != 'Queue':
        jsonable_config[k] = v

    # log 位置: build/ccsrc/policy/include/policy/aware/
    #           ptb_10880_myradio_4models/eval-dir/ptb/100843
    with open('%s/config.txt' % (self.eval_dir), 'w') as f:
      f.write(' '.join(sys.argv) + '\n')
      f.write('PID: %d\n' % os.getpid())
      json.dump(jsonable_config, f, indent = 4, sort_keys = True)

  def startup_strategy(self, pickle_input : Pickle):
    self.nx_graph = copy.deepcopy(pickle_input.get_nx_graph())

    start_time = time.time()
    self.dataset = os.path.split(self.pickled_inp_file)
    aware_simuator = AwareSimulator(pickle_input = pickle_input, 
                                    n_devs = self.n_devs)
    progress_placer = ProgressivePlacer(self.nx_graph, self, aware_simuator)
            
    # ProgressivePlacer().place(G, n_devices, nn_model, config, pptf)

