import os
import sys
import json
from collections import deque
import multiprocessing as mp
import random
import copy
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy
import networkx as nx
from itertools import count
from python.py_input import *
from python.progressive_nn import *
from python.progressive_graph import *
from tensorflow.python.framework import ops as tf_ops

class ProgressivePlacer:
  def __init__(self, nx_graph : nx.DiGraph(), config, aware_simuator):
    self.id = config.id
    self.m_name = config.m_name
    self.seed = self.id + config.seed
    self.set_seeds(self.seed)
    self.num_children = config.num_children
    self.n_peers = config.n_peers
    self.n_devs = config.n_devs
    # self.sim_eval = sim_eval
    self.disc_fact = config.disc_factor
    self.n_episodes = config.n_eps
    self.max_rounds = config.max_rnds
    self.print_freq = config.print_freq
    self.discard_last_rnds = config.discard_last_rnds

    self.tb_dir = config.tb_dir
    self.eval_dir = config.eval_dir
    self.fig_dir = config.fig_dir
    self.record_best_pl_file = config.record_best_pl_file
    self.tb_log_freq = 2

    self.eval_freq = config.eval_freq
    self.m_save_path = config.m_save_path
    self.restore_from = config.restore_from
    self.best_runtimes = []
    self.n_max_best_runtimes = 5
    self.record_pl_write_freq = 1
    self.ep2pl = {}
    self.debug_verbose = config.debug_verbose
    self.dont_share_classifier = config.dont_share_classifier
    self.eval_on_transfer = config.eval_on_transfer
    self.dont_repeat_ff = config.dont_repeat_ff
    self.gen_profile_timeline = config.gen_profile_timeline
    self.profiling_chrome_trace = config.profiling_chrome_trace
    self.node_traversal_order = config.node_traversal_order
    self.cache_eval_plts = deque(maxlen = 5)
    self.radial_mp = config.radial_mp
    self.one_shot_episodic_rew = config.one_shot_episodic_rew
    self.dont_restore_softmax = config.dont_restore_softmax
    self.save_freq = config.save_freq
    self.nx_graph = nx_graph
    self.aware_simuator = aware_simuator

    if self.gen_profile_timeline:
      self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      self.run_metadata = tf.RunMetadata()

    if self.one_shot_episodic_rew:
      assert config.zero_placement_init
      assert not config.use_min_runtime
    
    self.async_sim = (config.n_async_sims is not None) or \
                     (config.remote_async_addrs is not None)

    self.progressive_graph = ProgressiveGraph(self.nx_graph, self.n_devs, 
      self.node_traversal_order, self.radial_mp, seed=self.seed)
    
    if self.async_sim:
      self.setup_async_sim(config)
    
    if self.max_rounds is None:
      self.max_rounds = self.progressive_graph.n_nodes()
    
    self.nn_model = self.choose_model()
    self.model = self.nn_model(emb_size = self.progressive_graph.get_embed_size(),
                               n_nodes = self.progressive_graph.n_nodes(),
                               progressive_graph = self.progressive_graph,
                               config = config)

    rnd_cum_rewards = [deque(maxlen = config.bl_n_rnds) for _ in range(self.max_rounds)]

    if config.log_tb_workers:
      self.tb_writer = tf.summary.FileWriter(self.tb_dir, flush_secs = 30)
      summ_names = ['run_times/episode_end_rt', 'run_times/best_so_far', 'run_times/best_rew_rt',
                    'ent/tanhc_const', 'run_times/ep_best_rt']
      
      if not config.dont_sim_mem:
        summ_names += ['run_times/rew_rt', 'mem/mem_util', 'mem/best_mem_util_so_far']
      
      if config.supervised:
        summ_names += ['loss/loss', 'opt/logits', 'opt/lr']
      else:
        summ_names += ['rew/reward', 'loss/loss', 'rew/baseline', 'rew/advantage', 'loss/log_probs', \
                       'opt/lr', 'ent/entropy', 'opt/grad_norm', 'ent/ent_dec', 'opt/pre_sync_grad_norm']
    
    eval_summ_data = []
    self.save_saver = tf.train.Saver(max_to_keep = 100, keep_checkpoint_every_n_hours = 2)

    variables = tf_ops.get_collection(tf_ops.GraphKeys.TRAINABLE_VARIABLES)
    if self.dont_restore_softmax:
      variables = list(filter(lambda k: 'classifier' not in k.name, variables))
      self.restore_saver = tf.train.Saver(variables)
    else:
      self.restore_saver = tf.train.Saver()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    with tf.Session(config = sess_config) as sess:
      self.initialize_weights(sess)
      zero_placement = self.progressive_graph.get_zero_placement()
      rand_placement = self.progressive_graph.get_random_placement(seed = 0)
      null_placement = self.progressive_graph.get_null_placement()

      # 先给最佳的奖励执行时间和最佳放置分别设置为int最大值和随机放置
      best_rew_rt = 1e9
      best_pl = rand_placement
         
      disamb_nodes = []
      if config.disamb_pl:
        n = self.progressive_graph.n_nodes()
        disamb_nodes = ['0', '1', str(int((n / 2) - 1)), str(n - 1)]
        for i in disamb_nodes:
          rand_placement[str(i)] = 0
      
      nodes = [node for node in self.progressive_graph.nodes() if node not in disamb_nodes]

      # 开始训练
      for episode in range(self.n_episodes):
        if config.vary_init_state:
          self.init_pl = self.progressive_graph.get_random_placement(seed = ep)
        elif config.init_best_pl:
          self.init_pl = best_pl
        elif config.zero_placement_init:
          self.init_pl = zero_placement
        elif config.null_placement_init:
          self.init_pl = null_placement
        else:
          self.init_pl = rand_placement

        if self.eval_on_transfer is not None:
          is_eval_on_transfer = (episode == self.eval_on_transfer)
        else:
          is_eval_on_transfer = False
        
        is_eval_episode = (episode % self.eval_freq == 0) or is_eval_on_transfer
        is_save_episode = (episode % self.save_freq == 0 and episode > 0)
      
        _, episode_best_run_time, episode_best_mem_util, run_times, mem_utils, states, explor_acts, pls = \
          self.run_episode(sess, self.init_pl, nodes, is_eval_episode, episode, config)
      
      

  def run_episode(self, sess, init_pl, nodes, is_eval_episode, episode, config):
    self.progressive_graphs = [ProgressiveGraph(self.nx_graph, self.n_devs, self.node_traversal_order, 
      self.radial_mp, seed = self.seed) for _ in range(self.num_children)]

    for progressive_graph in self.progressive_graphs:
      progressive_graph.reset_placement(init_pl) # 将所有的设备都设置成init_pl
      progressive_graph.new_episode() # 将所有的节点和当前节点的flag标记为unseen

    start_times = [np.array([[-1] * progressive_graph.n_nodes()]) 
      for progressive_graph in self.progressive_graphs]

    if config.one_shot_episodic_rew:
      run_time = [math.inf] * self.num_children
      mem_util = [[math.inf] * self.n_devs] * self.num_children
    else:
      if self.async_sim:
        run_time, _, mem_util = self.eval_placement()
      else:
        run_time, start_times, mem_util = self.eval_placement()
    
      for i, progressive_graph in enumerate(self.progressive_graphs):
        progressive_graph.set_start_times(start_times[i])

    if (1 + episode) % self.print_freq == 0:
      print("Run time", run_time, end=' ')

      nx_graph = self.progressive_graphs[0].nx_graph
      pos = nx.kamada_kawai_layout(nx_graph)

      groups = set(range(config.n_devs))
      nodesToPrint = nx_graph.nodes
      mappingToPrint = dict(zip(sorted(groups), count()))
      colorsToPrint = [mappingToPrint[nx_graph.nodes[n]['placement']] for n in nodesToPrint]
      [print(n + ' ' + str(nx_graph.nodes[n]['placement'])) for n in nx_graph.nodes]

      with open('%s/graph_data_11.pkl' % self.fig_dir, 'wb') as file:
        pickle.dump({"Gg": nx_graph, "nodesToPrint": nodesToPrint, "colorsToPrint": colorsToPrint, "pos" :pos}, file)

      nc = nx.draw_networkx(nx_graph, pos,
                            nodelist=nodesToPrint,
                            node_color=colorsToPrint,
                            with_labels=False,
                            node_size=10,
                            font_size=3,
                            width=.5,
                            arrowsize=5,
                            cmap='plasma')
      # TODO remove for remote play
      # plt.colorbar(nc)
      plt.axis('off')

      placement_graph_img_location = '%s/graph-%d.pdf' % (self.fig_dir, episode)
      plt.savefig(placement_graph_img_location, dpi = 300)
      plt.clf()
      print('Saved graph placement image at %s' % placement_graph_img_location)
    
    episode_best_time = min(run_time)
    episode_best_pl_mem = mem_util[run_time.index(min(run_time))]
    episode_best_pl = init_pl
    run_times = []
    mem_utils = []
    states = []
    explor_acts = []
    async_record = []
    pls = [[init_pl] for _ in range(self.num_children)]

    run_times.append(run_time)
    mem_utils.append(mem_util)

    nn_time = 0
    s1 = time.time()
    for i in range(self.max_rounds):
      is_last_rnd = (i == self.max_rounds - 1)
      node = nodes[i % len(nodes)]
      for progressive_graph in self.progressive_graphs: 
        progressive_graph.set_curr_node(node)

      s2 = time.time()
      d, lo, feed, expl, train_outs = self.get_improvement(
        sess, node, start_times, is_eval_episode)
      nn_time += (time.time() - s2)

      explor_acts.append(expl)
      for j, progressive_graph in enumerate(self.progressive_graphs):
        progressive_graph.refresh([node], [d[j]])

      for j, progressive_graph in enumerate(self.progressive_graphs):
        pls[j].append(progressive_graph.get_placement())

      if not config.one_shot_episodic_rew or is_last_rnd:
        if self.async_sim:
          j = i % self.n_async_sims
          self.eval_placement(asynch = j)
          async_record.append(j)
        else:
          run_time, start_times, mem_util = self.eval_placement()
          for st, progressive_graph in zip(start_times, self.progressive_graphs):
            progressive_graph.update_start_times(st)

      # add infs if one shot
      if not self.async_sim or config.one_shot_episodic_rew:
        run_times.append(run_time)
        mem_utils.append(mem_util)

      states.append([feed, d, lo, train_outs])
      for progressive_graph in self.progressive_graphs:
        progressive_graph.inc_done_node(node)

    if self.async_sim:
      for j in async_record:
        run_time, mem_util = self.eval_placement(retreive = j)
        run_times.append(run_time)
        mem_utils.append(mem_util)

    for i, rnd_rt in enumerate(run_times):
      for j, rt in enumerate(rnd_rt):
        if episode_best_time > rt:
          episode_best_time = rt
          episode_best_pl_mem = mem_utils[i][j]
          episode_best_pl = pls[j][i]

    run_times = np.transpose(run_times)
    mem_utils = np.array(mem_utils)
    mem_utils = mem_utils.transpose(1, 0, 2)

    if episode < 20:
      print('Total time: ', time() - s1)
      print('NN time: ', nn_time)

    return episode_best_pl, episode_best_time, episode_best_pl_mem, run_times, mem_utils, states, explor_acts, pls

  def get_improvement(self, sess, node, start_times, is_eval_episode):
    model = self.model
    feed = model.get_feed_dict(self.progressive_graphs, node, start_times, self.n_peers)
    if is_eval_episode:
      feed[model.is_eval_ph] = 1.0

    train_ops = []
    if self.dont_repeat_ff:
      train_ops = [model.logprob_grad_outs, model.ent_grad_outs, \
                   model.log_probs, model.sample, \
                   model.pl_ent_loss, model.log_prob_loss, \
                   model.no_noise_classifier, model.entropy, \
                   model.ent_dec]

    kwargs = {}
    if self.gen_profile_timeline:
      kwargs = {'run_metadata': self.run_metadata,
                'options': self.run_options}

    print("self.sample : type {}".format(type(model.get_eval_ops()[0])))
    print("self.classifier : type {}".format(type(model.get_eval_ops()[1])))
    print("self.log_probs : type {}".format(type(model.get_eval_ops()[2])))
    print("self.logprob_grad_outs : type {}".format(type(model.logprob_grad_outs)))
    print("self.ent_grad_outs : type {}".format(type(model.ent_grad_outs)))
    print("self.log_probs : type {}".format(type(model.log_probs)))
    print("self.sample : type {}".format(type(model.sample)))
    print("self.pl_ent_loss : type {}".format(type(model.pl_ent_loss)))
    print("self.log_prob_loss : type {}".format(type(model.log_prob_loss)))
    print("self.no_noise_classifier : type {}".format(type(model.no_noise_classifier)))
    print("self.entropy : type {}".format(type(model.entropy)))
    print("self.ent_dec : type {}".format(type(model.ent_dec)))

    
    s, lo, lp, expl, *train_outs = sess.run(model.get_eval_ops() + \
                                              [model.expl_act] + \
                                              train_ops,
                                              feed_dict = feed, **kwargs)

    return s, lo, feed, expl, train_outs

  def eval_placement(self, pl = None, asynch = None, retreive = None):
    pls = []
    if pl is None:
      for progressive_graph in self.progressive_graphs:
        pls.append(copy.copy(progressive_graph.get_placement()))
    else:
      pls.append(copy.copy(pl))

    if asynch is not None:
      self.async_send_pls_q[asynch].put(pls)
      return
    elif retreive is not None:
      return self.async_recv_pls_q[retreive].get()
    else:
      run_times = []
      start_times = []
      mem_utils = []
      for progressive_graph, pl in zip(self.progressive_graphs, pls):
        run_time, start_time, mem_util = self.aware_simuator.simulate(pl)
        run_times.append(run_time)
        start_times.append(start_time)
        mem_utils.append(mem_util)
    
    return run_times, start_times, mem_utils

  def choose_model(self):
    if self.m_name == 'simple_nn':
      nn_model = SimpleNN
    elif self.m_name == 'mp_nn':
      nn_model = MessagePassingProgressiveNN
    else:
      raise Exception('%s not implemented model' % self.m_name)
    return nn_model
    
  def setup_async_sim(self, config):
    self.async_send_pls_q = []
    self.async_recv_pls_q = []

    if config.n_async_sims:
      for i in range(config.n_async_sims):
        self.async_send_pls_q.append(mp.Queue(1000))
        self.async_recv_pls_q.append(mp.Queue(1000))

        d = {'id': i,
             'recv_q': self.async_send_pls_q[-1],
             'send_q': self.async_recv_pls_q[-1],
             'nx_graph': self.progressive_graphs.get_nx_graph()}
        mp.Process(target = self.async_process_func, args = (d,)).start()

    self.n_async_sims = len(self.async_send_pls_q) 

  def set_seeds(self, i : int = 0):
    if i is None:
        i = 0
    s = 42 + i
    np.random.seed(s)
    tf.set_random_seed(s)
    random.seed(s)

  def initialize_weights(self, sess):
    if self.restore_from is not None:
      if self.dont_restore_softmax:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
      self.restore_saver.restore(sess, self.restore_from)
      print('Model successfully restored from "%s"!' % self.restore_from)
    else:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

    sess.run(self.model.init_global_step)
