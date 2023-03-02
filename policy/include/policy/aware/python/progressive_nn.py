from python.progressive_fnn import *
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.training import adam, rmsprop, gradient_descent
import tensorflow as tf

class ReinforceAgent(object):
  def __init__(self, seed : int, reinforce_params : dict):
    self.seed : int = seed
    self.bs = 1
    self.global_step = tf.train.get_or_create_global_step()
    self.init_global_step = tf.assign(self.global_step, 0)
    self.is_eval_ph = tf.placeholder_with_default(0., None)

    self.lr_init : float = reinforce_params.get('lr_init', None)
    self.lr_dec : float = reinforce_params.get('lr_dec', None)
    self.lr_start_decay_step : int = reinforce_params.get('lr_start_decay_step', None)
    self.lr_decay_steps : int = reinforce_params.get('lr_decay_steps', None)
    self.lr_min : float = reinforce_params.get('lr_min', None)
    self.lr_dec_approach : str = reinforce_params.get('lr_dec_approach', None)
    self.ent_dec_init : float = reinforce_params.get('ent_dec_init', None)
    self.ent_dec : float = reinforce_params.get('ent_dec', None)
    self.ent_start_dec_step : int = reinforce_params.get('ent_start_dec_step', None)
    self.ent_dec_steps : int = reinforce_params.get('ent_dec_steps', None)
    self.ent_dec_min : float = reinforce_params.get('ent_dec_min', None)
    self.ent_dec_lin_steps : int = reinforce_params.get('ent_dec_lin_steps', None)
    self.ent_dec_approach : str = reinforce_params.get('ent_dec_approach', None)
    self.optimizer_type : str = reinforce_params.get('optimizer_type', None)
    self.eps_init : float = reinforce_params.get('eps_init', None)
    self.eps_dec_steps : int = reinforce_params.get('eps_dec_steps', None)
    self.start_eps_dec_step : int = reinforce_params.get('start_eps_dec_steps', None)
    self.stop_eps_dec_step : int = reinforce_params.get('stop_eps_dec_step', None)
    self.eps_dec_rate : float = reinforce_params.get('eps_dec_rate', None)
    self.tanhc_init : float = reinforce_params.get('tanhc_init', None)
    self.tanhc_dec_steps : int = reinforce_params.get('tanhc_dec_steps', None)
    self.tanhc_max : float = reinforce_params.get('tanhc_max', None)
    self.tanhc_start_dec_step : int = reinforce_params.get('tanhc_start_dec_step', None)
    self.no_grad_clip : bool = reinforce_params.get('no_grad_clip', None)

    self.lr_start_decay_step = int(self.lr_start_decay_step)
    self.lr_gstep : int = self.global_step - self.lr_start_decay_step

    self.ent_start_dec_step = int(self.ent_start_dec_step)
    ent_gstep = self.global_step - self.ent_start_dec_step
    if self.ent_dec_approach == 'exponential':
        self.ent_dec_func = tf.train.exponential_decay(self.ent_dec_init, 
                        ent_gstep, self.ent_dec_steps, self.ent_dec, False),
    elif self.ent_dec_approach == 'linear':
        self.ent_dec_func = tf.train.polynomial_decay(self.ent_dec_init, 
                        ent_gstep, self.ent_dec_lin_steps, self.ent_dec_min)
    elif self.ent_dec_approach == 'step':
        self.ent_dec_func = tf.constant(self.ent_dec_min)
    ent_dec = tf.cond(
          tf.less(self.global_step, self.ent_start_dec_step),
          lambda: tf.constant(self.ent_dec_init),
          lambda: self.ent_dec_func,
          name = 'ent_decay')    
    self.ent_dec = tf.maximum(ent_dec, self.ent_dec_min)

  def _sample(self, classifier):
    sample_argmax = tf.argmax(classifier, axis = -1)
    sample = tf.multinomial(classifier, 1, seed = self.seed)
    sample = tf.reshape(tf.cast(sample, tf.int32), [-1])
    sample_argmax = tf.reshape(tf.cast(sample_argmax, tf.int32), [-1])
    # use during eval phase
    expl_act = tf.logical_not(tf.equal(sample, sample_argmax))
    log_probs = -1. * tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits = classifier, labels = sample)

    return sample, log_probs, expl_act
  
  def _get_entropy(self, classifier):
    # with tf.name_scope('Entropy_logits'):
    p = tf.nn.softmax(classifier)
    lp = tf.math.log(p + 1e-3)
    entropy = - p * lp
    entropy = tf.reduce_sum(entropy, axis = -1)
    return entropy

  def setup_lr(self):
    if self.lr_dec_approach == 'linear':
      learning_rate = tf.cond(
        pred = tf.less(self.global_step, self.lr_start_decay_step),
        true_fn = self.constant_func,
        false_fn = self.poly_func,
        name="learning_rate")
    else:
      learning_rate = tf.cond(
        pred = tf.less(self.global_step, self.lr_start_decay_step),
        true_fn = self.constant_func,
        false_fn = self.exp_func,
        name = "learning_rate")

    self.lr = tf.maximum(learning_rate, self.lr_min)
  
  def constant_func(self):
    return tf.constant(self.lr_init)
  
  def exp_func(self):
    return tf.train.exponential_decay(self.lr_init, self.lr_gstep,
             self.lr_decay_steps, self.lr_dec, True)
  
  def poly_func(self):
    return tf.train.polynomial_decay(self.lr_init, self.lr_gstep,
             self.lr_decay_steps, self.lr_min)

  def _get_optimizer(self):
    self.setup_lr()
    # tf.summary.scalar('lr', self.lr)
    optimizer_type = self.optimizer_type
    if optimizer_type == "adam":
      opt = adam.AdamOptimizer(self.lr)
    elif optimizer_type == "sgd":
      opt = gradient_descent.GradientDescentOptimizer(self.lr)
    elif optimizer_type == "rmsprop":
      opt = rmsprop.RMSPropOptimizer(self.lr)
    return opt

  def _build_train_ops(self, grad_bound = 1.25, dont_repeat_ff = False):
    tf_variables = tf_ops.get_collection(tf_ops.GraphKeys.TRAINABLE_VARIABLES),
    opt = self._get_optimizer()

    # print some ent, adv stats
    all_grads = []
    b_grads = []
    for i in range(self.bs):
      with tf.variable_scope('log_prob_grads'):
        grads_and_vars = opt.compute_gradients(self.log_prob_loss[i], tf_variables)
      b_grads.append(grads_and_vars)
      for x in grads_and_vars:
        all_grads.append(x)

    grad_norm = clip_ops.global_norm([tf.cast(g, tf.float32) for g, _ in all_grads if g is not None])
    self.logprob_grad_outs = [[g for g, _ in b_grads[i] if g is not None] for i in range(self.bs)]

    # print some ent, adv stats
    all_grads2 = []
    b_grads2 = []
    for i in range(self.bs):
      with tf.variable_scope('placement_ent_grads'):
        grads_and_vars2 = opt.compute_gradients(self.pl_ent_loss[i], tf_variables)
      b_grads2.append(grads_and_vars2)
      for x in grads_and_vars2:
        all_grads2.append(x)

    grad_norm2 = clip_ops.global_norm([tf.cast(g, tf.float32) for g, _ in all_grads2 if g is not None])
    self.ent_grad_outs = [[g for g, _ in b_grads2[i] if g is not None] for i in range(self.bs)]

    self.reinforce_grad_norm = tf.reduce_mean(grad_norm)
    self.entropy_grad_norm = tf.reduce_mean(grad_norm2)
    self.grad_phs = []
    self.grad_outs = []
    gradphs_and_vars = []

    # if not dont_repeat_ff:
    # grads_and_vars = opt.compute_gradients(loss, tf_variables)
    self.grad_outs = None
    for i, [g, v] in enumerate(grads_and_vars):
      if g is not None:
        # if not dont_repeat_ff: 
        # self.grad_outs.append(g)
        grad_vtype = tf.float32
        if v.dtype == tf.as_dtype('float16_ref'):
          grad_vtype = tf.float16
        p = tf.placeholder(grad_vtype, name='grad_phs_%d' % i)
        self.grad_phs.append(p)
        gradphs_and_vars.append((p, v))

    self.grad_norm = tf.linalg.global_norm([tf.cast(g, tf.float32) for g in self.grad_phs])
    self.gradphs_and_vars = gradphs_and_vars
    
    if self.no_grad_clip:
      clipped_grads = gradphs_and_vars
    else:
      clipped_grads = self._clip_grads_and_vars(gradphs_and_vars, 
                                                  self.grad_norm, grad_bound)
    train_op = opt.apply_gradients(clipped_grads, self.global_step)

    return train_op, self.grad_outs, self.logprob_grad_outs, self.ent_grad_outs

  def _clip_grads_and_vars(self, grads_and_vars, grad_norm, grad_bound):
    all_grad_norms = {}
    clipped_grads = []
    clipped_rate = tf.maximum(grad_norm / grad_bound, 1.0)

    for g, v in grads_and_vars:
      if g is not None:
        if isinstance(g, tf_ops.IndexedSlices):
          raise Exception('IndexedSlices not allowed here')
        else:
          clipped = g / tf.cast(clipped_rate, g.dtype)
          norm_square = tf.reduce_sum(clipped * clipped, axis = -1)

        all_grad_norms[v.name] = tf.sqrt(norm_square)
        clipped_grads.append((clipped, v))

    return clipped_grads

class ProgressiveNN(ReinforceAgent):
  def __init__(self, seed = 42, dont_repeat_ff = False, reinforce_params : dict = {}):
    ReinforceAgent.__init__(self, seed, reinforce_params)
    self.dont_repeat_ff = dont_repeat_ff

  def build_train_ops(self, classifier):
    self.classifier = classifier
    self.sample, self.log_probs, self.expl_act = self._sample(classifier)
    self.entropy = self._get_entropy(classifier)
    # note that loss resamples instead of reading from ph
    self.pl_ent_loss = - self.entropy * self.ent_dec
    self.log_prob_loss = - self.log_probs
    self.train_op, self.grad_outs, self.logprob_grad_outs, self.ent_grad_outs = \
      self._build_train_ops(dont_repeat_ff = self.dont_repeat_ff)

  def get_eval_ops(self):
    return [self.sample, self.classifier, self.log_probs]

class MessagePassingProgressiveNN(ProgressiveNN):
  def __init__(self, emb_size, n_nodes, progressive_graph, config):
    ProgressiveNN.__init__(self, config.seed, config.dont_repeat_ff, config.reinforce_params)

    self.emb_size = emb_size
    self.n_nodes = n_nodes
    self.progressive_graph = progressive_graph
    self.config = config
    self.input_tensor = tf.placeholder(tf.float32)
    self.bs = 1

    self.radio_messenger = RadioMessenger(radial_mp = config.radial_mp,
                                          embedding_size_deg = self.emb_size,
                                          embedding_transformation_deg = self.emb_size,
                                          sage_sample_ratio = config.sage_sample_ratio,
                                          sage_hops = config.sage_hops,
                                          sage_aggregation = config.sage_aggregation,
                                          sage_dropout_rate = config.sage_dropout_rate,
                                          position_aware = config.sage_position_aware,
                                          pgnn_c = config.pgnn_c,
                                          pgnn_neigh_cutoff = config.pgnn_neigh_cutoff,
                                          pgnn_anchor_exponent = config.pgnn_anchor_exponent,
                                          pgnn_aggregation = config.pgnn_aggregation,
                                          dtype = tf.float32)
    
    output = self.radio_messenger.build(self.progressive_graph, self.input_tensor)
    
    self.aggregator_size = 2 * self.emb_size
    
    args = [self.aggregator_size, self.aggregator_size, 
            self.aggregator_size, True, config.normalize_aggs]
    with tf.variable_scope('Parent-Aggregator'):
      self.agg_p = Aggregator(*args)
      agg_p_out = self.agg_p.build(output)
    with tf.variable_scope('Child-Aggregator'):
      self.agg_c = Aggregator(*args)
      agg_c_out = self.agg_c.build(output)
    with tf.variable_scope('Parallel-Aggregator'):
      self.agg_r = Aggregator(*args)
      agg_r_out = self.agg_r.build(output)
    with tf.variable_scope('Self-Embedding'):
      self.self_mask = tf.placeholder(tf.float32, [self.bs, None])
      self_out = tf.matmul(self.self_mask, output)
    self.agg_p_out, self.agg_c_out, self.agg_r_out = agg_p_out, agg_c_out, agg_r_out
    self.self_out = self_out
    self.triagg_out = [agg_p_out, agg_c_out, agg_r_out, self_out]
    
    output = tf.reshape(output, [self.bs, -1])
    input_size = output.get_shape()[-1]

    if config.bn_pre_classifier:
      output = tf.layers.batch_normalization(output, training=True)
    
    classifier_hidden_layers = [2 * input_size, input_size]
    classifier = Classifier(input_size, classifier_hidden_layers, config.n_devs).build(output)
    self.no_noise_classifier = classifier
    self.build_train_ops(classifier)

  def get_feed_dict(self, progressive_graphs, node, start_times, n_peers):
    if len(progressive_graphs) == 1:
      input_tensor = progressive_graphs[0].get_embeddings()
    else:
      input_tensor = []
      for progressive_graph in progressive_graphs:
        E.append(progressive_graphs.get_embeddings())
    d = {self.input_tensor: input_tensor}
    bs = len(progressive_graphs)
    node_num = progressive_graphs[0].n_nodes()
      
    p_masks = self.get_mask([progressive_graph.get_ancestral_mask 
                              for progressive_graph in progressive_graphs], 
                              bs, node_num, node)
    c_masks = self.get_mask([progressive_graph.get_progenial_mask 
                              for progressive_graph in progressive_graphs],
                              bs, node_num, node)
    r_masks = self.get_mask([
              lambda node: progressive_graph.get_peer_mask(node, start_t, n_peers)
              for start_t, progressive_graph in zip(start_times, progressive_graphs)
              ], bs, node_num, node)
    self_masks = self.get_mask([progressive_graph.get_self_mask 
                                for progressive_graph in progressive_graphs],
                                bs, node_num, node)

    d = {
        self.input_tensor: np.array(input_tensor),
        self.agg_p.get_ph(): np.array(p_masks),
        self.agg_c.get_ph(): np.array(c_masks),
        self.agg_r.get_ph(): np.array(r_masks),
        self.get_ph(): np.array(self_masks)
        }
      
    return d

  def get_ph(self):
    return self.self_mask

  def get_mask(self, mask_fns, bs, node_num, node):
    mask = np.zeros((bs, bs * node_num), dtype = np.int32)
    for i in range(bs):
      mask[i, i * node_num:(i + 1) * node_num] = mask_fns[i](node)
    return mask

class SimpleNN(ProgressiveNN):
  def __init__(self, emb_size, n_nodes, progressive_graph, config):
    ProgressiveNN.__init__(self, seed, config.dont_repeat_ff)

    self.n_devs = config.n_devs
    self.E = tf.placeholder(tf.float32)
    inp_size = emb_size * n_nodes
    self.fnn = FNN(inp_size, [2 * inp_size, 2 * inp_size], n_devs, 'simple_nn')
    self.no_noise_logits = self.fnn.build(E)
    self.build_train_ops(self.no_noise_logits)

  def get_feed_dict(self, pg, node, start_times, n_peers):
    E = pg.get_embeddings().flatten()
    E = np.expand_dims(E, axis=0)
    d = {self.E: E}
    return d
