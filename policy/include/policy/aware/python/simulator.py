from collections import defaultdict
import heapq

class Node(object):
  def __init__(self):
    self.op_name = None
    self.device = None
    self.compute_cost = None
    self.output_memory = None
    self.parents = None
    self.children = None

class SimQueue(object):
  def __init__(self):
    self.queue = []

  def put(self, x):
    heapq.heappush(self.queue, x)

  def get(self):
    return heapq.heappop(self.queue)

  def empty(self):
    return len(self.queue) == 0

class Simulator(object):
  def __init__(self, meta_graph, cost_dict, output_dict, dev_names):
    self.meta_graph = meta_graph
    self.cost_dict = defaultdict(int, cost_dict)
    self.output_dict = defaultdict(list, output_dict)
    self.bus = "/bus"
    self.dev_names = dev_names
    self.node_dict = self.get_attributes()
    self.params = {
      "delta1" : 5.7,
      "delta2" : 25,
      "init_offset" : 0,
      "transfer_speed" : 7600
    }

    # Make a parent_map : node_name -> bool map of parents
    self.parent_map = defaultdict(dict)
    for k, v in self.node_dict.items():
      for p in v.parents:
        self.parent_map[k][p] = True

  def add_to_dev_queue(self, t, op, dev, element):
    self.count += 1
    self.device_queue[dev].put(((t, self.count), element))
    if not self.device_in_queue[dev]:
      self.sim_queue.put((t, op, dev))
      self.device_in_queue[dev] = True

  # Run bus
  def run_bus(self, t, dev):
    p, (node_name, delay, child_list) = self.device_queue[dev].get()
    # If bus is scheduled to run later, run later
    if p[0] > t:
      self.device_queue[dev].put((p, (node_name, delay, child_list)))
      self.sim_queue.put((p[0], "run_bus", dev))
      return
    for c in child_list:
      self.sim_queue.put((t + delay, "remove_dependency", (node_name, c)))
    self.sim_queue.put((t + delay + self.params["delta2"], "run_bus", dev))   

  # Runs the next job on device
  def run_dev(self, t, dev):
    p, node_name = self.device_queue[dev].get()
    node = self.node_dict[node_name]
    # Compute start and end times

    # f[node_name] = self.Node()
    self.node_dict[node_name].start_time = t
    self.node_dict[node_name].end_time = t + node.compute_cost
    self.node_dict[node_name].device = dev

    # Schedule when device is free again
    delta = 0 if node.compute_cost == 0 else self.params["delta1"]
    self.sim_queue.put((self.node_dict[node_name].end_time + delta, "run_dev", dev))
      
    # Find which all output indices require bus
    require_bus = defaultdict(list) # output_index to list of children
    for c, o in node.children.items():
      if dev == self.get_dev(c):
        self.sim_queue.put((self.node_dict[node_name].end_time, "remove_dependency", (node_name, c)))
      else:
        require_bus[o].append(c)
      
    # Schedule transfer on bus
    for o, c_list in require_bus.items():
      delay = node.output_memory[o] / self.params["transfer_speed"]
      self.add_to_dev_queue(self.node_dict[node_name].end_time, "run_bus", self.bus + dev, (node_name, delay, require_bus[o]))
  
  def get_dev(self, k):
    if k in self.device_dict:
      return self.device_dict[k]
    else:
      print('not in device_dict ', k)
      raise Exception('device not assigned for op %s' % k)
      return self.node_dict[k].device

  def remove_dependency(self, t, parent_name, child_name):
    self.parent_map[child_name][parent_name] = False
    # Schedule child if no more dependencies
    if self.is_scheduleable(child_name):
      self.add_to_dev_queue(t, "run_dev", self.get_dev(child_name), child_name)  

  def is_scheduleable(self, n):
    for v in self.parent_map[n].values():
      if v: return False
    return True

  def simulate(self, device_dict : dict):
    self.count, run_time = 0, 0
    self.sim_queue = SimQueue()
    all_devs = self.dev_names + [self.bus + dev for dev in self.dev_names]
    self.device_in_queue = dict((dev, False) for dev in all_devs)
    self.device_queue = dict((dev, SimQueue()) for dev in all_devs)
    self.device_dict = device_dict

    # Reset parent_map
    for k,v in self.parent_map.items():
      for p in v.keys():
        v[p] = True

    # Insert all runnable ops to device_queue
    for name, node in self.node_dict.items():
      if not node.parents:
        self.add_to_dev_queue(self.params["init_offset"], "run_dev", self.get_dev(name), name)

    # Main loop
    while not self.sim_queue.empty():
      t, op, dev = self.sim_queue.get()
      run_time = max(run_time, t)
      if (op == "run_bus" or op == "run_dev") and self.device_queue[dev].empty():
        self.device_in_queue[dev] = False
        continue
      elif op == "run_bus":
        self.run_bus(t, dev)
      elif op == "remove_dependency":
        p_name, c_name = dev
        self.remove_dependency(t, p_name, c_name)
      elif op == "run_dev":
        self.run_dev(t, dev)

    return run_time, self.node_dict

  def get_attributes(self):
    """ 
      Creates the node_dict. Node contains the following
      Attributes
        op_name: name of op
        device: device of the node
        compute_cost: run time in ns
        output_memory: list of output sizes
        parents: set of parents
        children: dict from node_name to output_index
    """
    # Create a dict from node_name -> Node
    node_dict = dict()

    # Set default values
    for node in self.meta_graph.graph_def.node:
      node_dict[node.name] = Node()
      node_dict[node.name].op_name = node.op
      node_dict[node.name].device = node.device
      node_dict[node.name].compute_cost = self.cost_dict[node.name]
      node_dict[node.name].output_memory = self.output_dict[node.name]
      node_dict[node.name].parents = set()
      node_dict[node.name].children = dict()
    
    # iterate through all the nodes of the graph
    for node in self.meta_graph.graph_def.node:
      # If neither CPU or GPU, then put on CPU
      for i in node.input:
        i = i[1:] if i[0] == '^' else i
        i = (i + ":0") if ":" not in i else i
        i = i.split(":")
        # Set parents and children
        parent, out_idx = i[0], int(i[1])
        while out_idx >= len(node_dict[parent].output_memory):
          node_dict[parent].output_memory.append(0)
        node_dict[node.name].parents.add(parent)
        node_dict[parent].children[node.name] = out_idx

    return node_dict

class ImportantOpsSimulator(Simulator):
  def __init__(self, meta_graph, op_perf, step_stats, dev_names):
    cost_d, _ = self.get_op_costs(step_stats)

    out_d = {}
    for op in op_perf:
      out_d[op.node] = op.op_memory.output_memory

    for dev_stats in step_stats.dev_stats:
      for node_stats in dev_stats.node_stats:
        node = node_stats.node_name
        for output in node_stats.output:
          allocation = output.tensor_description.allocation_description
          num_bytes = allocation.requested_bytes
          out_d[node] = [num_bytes]
          break

    for i, dev in enumerate(dev_names):
      dev_names[i] = '/' + dev.split('/')[-1]
    for node in meta_graph.graph_def.node:
      d = node.device
      node.device = '/' + d.split('/')[-1]
    
    Simulator.__init__(self, meta_graph, cost_d, out_d, dev_names)

  def get_op_costs(self, step_stats):
    d = {}
    cost_d = {}

    for dev_stat in step_stats.dev_stats:
      # https://github.com/tensorflow/tensorflow/blob/4595f1cff635ce024e875f0f3d480172731b0b22/tensorflow/core/profiler/internal/tfprof_node.cc
      if 'all' in dev_stat.device: #or 'CPU' in dev_stat.device:
      # if 'cpu' not in dev_stat.device.lower():
        for node_stat in dev_stat.node_stats:
          n = node_stat.node_name.split(':')[0]
          if n not in d:
            d[n] = [node_stat.all_start_micros, node_stat.all_end_rel_micros - \
              node_stat.op_start_rel_micros]
          else:
            d[n][1] += node_stat.all_end_rel_micros - node_stat.op_start_rel_micros

          cost_d[n] = d[n][1]

    return cost_d, d

  def simulate(self, pl):
    for k, v in pl.items():
      pl[k] = self.dev_names[int(v)]
    
    runtime, node_dict = Simulator.simulate(self, pl)
    self.node_dict = node_dict

    start_times = {}
    for node in self.meta_graph.graph_def.node:
      n = node.name
      start_times[n] = node_dict[n].start_time

    mem_q = []
    for node, time in start_times.items():
      mem = sum(self.output_dict[node])
      if mem == 0:
        continue
      dev = self.dev_names.index(node_dict[node].device)
      mem_q.append((time, '+', mem, dev))
      t_out_done = time
      for c in node_dict[n].children:
        t_out_done = max(t_out_done, int(node_dict[c].start_time) + int(node_dict[c].compute_cost) - 1)
      mem_q.append((t_out_done, '-', - mem, dev))
    
    mem_q.sort()
    mem_utils = [0] * len(self.dev_names)
    peak_utils = [0] * len(self.dev_names)

    for (time, _, mem, dev) in mem_q:
      mem_utils[dev] += mem
      if mem_utils[dev] > peak_utils[dev]:
        peak_utils[dev] = mem_utils[dev]
    return runtime, start_times, peak_utils

class AwareSimulator(object):
  def __init__(self, pickle_input, n_devs):
    device_names = ['/device:GPU:%d' % i for i in range(n_devs)]
    gpu_devices = list(sorted(filter(lambda dev: 'GPU' in dev, device_names)))
    cost_d, out_d, temp_mem, mem_info = [None] * 4

    self.meta_graph = pickle_input.get_meta_graph()
    self.nx_graph = pickle_input.get_nx_graph()
    self.ungroup_mapping = pickle_input.get_ungrouped_mapping()
    self.op_perf = pickle_input.get_op_pref()
    self.step_stats = pickle_input.get_step_stats()
    self.n_devs = n_devs
    self.gpu_devs = gpu_devices
    self.cost_d = cost_d
    self.out_d = out_d
    self.dev_names = device_names

    self.simulator = ImportantOpsSimulator(self.meta_graph, 
                                          self.op_perf, 
                                          self.step_stats, 
                                          self.dev_names)
    
  def simulate(self, pl):
    for n in pl:
      pl[n] = str(pl[n])
    
    ungrouped_pl = self.ungroup_pl(pl)
    run_time, start_times, mem_utils = \
      self.simulator.simulate(ungrouped_pl)

    return run_time / 1e6, start_times, mem_utils
    
  def ungroup_pl(self, pl):
    ungrouped_pl = {}
    for op in self.meta_graph.graph_def.node:
      grp_ctr = self.ungroup_mapping[op.name]
      ungrouped_pl[op.name] = pl[grp_ctr]
    
    return ungrouped_pl

    


    

