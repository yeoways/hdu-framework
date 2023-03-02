import pickle

class PPTFItem(object):
  def __init__(self, pickled_inp_file, n_devs, simplify_tf_reward_model=False, 
               use_new_sim=False, sim_mem_usage=False):
    device_names = ['/device:GPU:%d' % i for i in range(n_devs)]
    gpu_devices = list(sorted(filter(lambda dev: 'GPU' in dev, device_names)))

    with open(pickled_inp_file, 'rb') as input_file:
      input_data = pickle.load(input_file)
      

    