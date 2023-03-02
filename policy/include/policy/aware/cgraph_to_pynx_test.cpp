#include <cgraph_to_pynx.h>
#include <pynx_to_cgraph.h>
#include <iostream>
#include <map>


int main(){
  // 总体流程, 调用 python 函数读取 pickle 文件中的计算图, 再转换成 cost_node
  // 的格式, 在 c++ 侧完成算子融合之后再将其送回 python 侧完成 embedding 和 reinforce
  // aware 的输入是一个 pickle 文件, 只能由 python 读取, 需要调用 python 函数完成
  py::scoped_interpreter python; //一定要加, 否则会报错

  py::dict py_dict;
  // 需要设置参数的项把注释消掉, 不需要设置注掉
  py_dict["seed"] = 42;
  py_dict["name"] = "test";
  // py_dict["graph"] = NULL;
  py_dict["id"] = 1;
  py_dict["graph_size"] = 4;
  py_dict["pickled_inp_file"] = "/mnt/e/DatasetWareHouse/datasets/ptb/100843/input.pkl";
  // py_dict["mul_graphs"] = NULL;
  py_dict["dataset_folder"] = "/mnt/e/DatasetWareHouse/datasets/ptb";
  py_dict["dataset"] = "ptb/100843";
  py_dict["n_devs"] = 4;
  py_dict["model_folder_prefix"] = "./ptb_10880_myradio_4";
  py_dict["m_name"] = "mp_nn";
  // py_dict["n_peers"] = NULL;
  // py_dict["agg_msgs"] = true;
  py_dict["no_msg_passing"] = false;
  py_dict["radial_mp"] = 1;
  // py_dict["tri_agg"] = true;
  // py_dict["sage"] = true;
  py_dict["sage_hops"] = 1;
  py_dict["sage_sample_ratio"] = 1.0;
  py_dict["sage_dropout_rate"] = 0.0;
  py_dict["sage_aggregation"] = "max";
  py_dict["sage_position_aware"] = true;
  // py_dict["use_single_layer_perceptron"] = false;
  py_dict["pgnn_c"] = 0.2;
  py_dict["pgnn_neigh_cutoff"] = 4;
  py_dict["pgnn_anchor_exponent"] = 4;
  py_dict["pgnn_aggregation"] = "max";
  // py_dict["reinit_model"] = NULL;
  py_dict["n_eps"] = 20;
  // py_dict["max_rnds"] = NULL;
  py_dict["disc_factor"] = 1.0;
  py_dict["vary_init_state"] = false;
  py_dict["zero_placement_init"] = false;
  py_dict["null_placement_init"] = false;
  py_dict["init_best_pl"] = true;
  py_dict["one_shot_episodic_rew"] = false;
  py_dict["ep_decay_start"] = 1000.0;
  py_dict["bl_n_rnds"] = 1000;
  py_dict["rew_singlegpu"] = false;
  py_dict["rew_neigh_pl"] = false;
  py_dict["supervised"] = true;
  py_dict["use_min_runtime"] = false;
  py_dict["discard_last_rnds"] = false;
  py_dict["turn_based_baseline"] = true;
  py_dict["dont_repeat_ff"] = true;
  py_dict["small_nn"] = false;
  py_dict["dont_restore_softmax"] = false;
  // py_dict["restore_from"] = NULL;
  py_dict["print_freq"] = 1;
  py_dict["save_freq"] = 1;
  py_dict["eval_freq"] = 5;
  py_dict["log_tb_workers"] = false;
  py_dict["debug"] = true;
  py_dict["debug_verbose"] = true;
  py_dict["disamb_pl"] = false;
  // py_dict["eval"] = NULL;
  // py_dict["simplify_tf_rew_model"] = false;
  py_dict["log_runtime"] = true;
  // py_dict["use_new_sim"] = true;
  py_dict["gen_profile_timeline"] = false;
  py_dict["mem_penalty"] = 3.0;
  py_dict["max_mem"] = 10.0;
  py_dict["max_runtime_mem_penalized"] = 10.0;
  py_dict["use_threads"] = false;
  py_dict["scale_norm"] = false;
  py_dict["dont_share_classifier"] = false;
  // py_dict["use_gpus"] = NULL;
  // py_dict["eval_on_transfer"] = NULL;
  py_dict["normalize_aggs"] = false;
  py_dict["bn_pre_classifier"] = false;
  // py_dict["bs"] = NULL;
  py_dict["num_children"] = 1;
  py_dict["disable_profiling"] = false;
  // py_dict["n_async_sims"] = NULL;
  // py_dict["baseline_mask"] = NULL;
  py_dict["n_workers"] = 1;
  py_dict["node_traversal_order"] = "random";
  // py_dict["prune_final_size"] = NULL;
  py_dict["dont_sim_mem"] = false;
  // py_dict["remote_async_addrs"] = NULL;
  // py_dict["remote_async_start_ports"] = NULL;
  // py_dict["remote_async_n_sims"] = NULL;
  // py_dict["local_prefix"] = NULL;
  // py_dict["remote_prefix"] = NULL;
  py_dict["shuffle_gpu_order"] = false;

  py::dict reinforce_params;
  reinforce_params["lr_init"] = 1e-3;
  reinforce_params["lr_dec"] = 0.95;
  reinforce_params["lr_start_decay_step"] = 1e9;
  reinforce_params["lr_decay_steps"] = 100;
  reinforce_params["lr_min"] = 1e-3;
  reinforce_params["lr_dec_approach"] = "exponential";
  reinforce_params["ent_dec_init"] = 1.0;
  reinforce_params["ent_dec"] = 0.95;
  reinforce_params["ent_start_dec_step"] = 1e9;
  reinforce_params["ent_dec_steps"] = 100;
  reinforce_params["ent_dec_min"] = 0.0;
  reinforce_params["ent_dec_lin_steps"] = 0;
  reinforce_params["ent_dec_approach"] = "linear";
  reinforce_params["optimizer_type"] = "adam";
  reinforce_params["eps_init"] = 0.0;
  reinforce_params["eps_dec_steps"] = 1e9;
  reinforce_params["start_eps_dec_step"] = 1e9;
  reinforce_params["stop_eps_dec_step"] = 1e9;
  reinforce_params["eps_dec_rate"] = 0.95;
  // reinforce_params["tanhc_init"] = None;
  // reinforce_params["tanhc_dec_steps"] = None;
  // reinforce_params["tanhc_max"] = None;
  reinforce_params["tanhc_start_dec_step"] = 0;
  reinforce_params["no_grad_clip"] = false;

  py::object pickle_input = py::module::import("python.py_input") \
                           .attr("Pickle")(py_dict["pickled_inp_file"]);
  
  ConvertPYNXToCMergedGraph pynx_to_merged_cgraph(pickle_input.attr("get_nx_graph")());
  pynx_to_merged_cgraph.convert_py_nx_graph();
  MergedCostGraph merged_cost_graph = pynx_to_merged_cgraph.get_merged_cost_graph();

  // 算子融合部分还没接上, 这部分补上整个流程就打通了

  ConvertCMergedGraphToPYNX merged_cgraph_to_pynx(merged_cost_graph);
  // 转换成nx_graph.py中的graph形式
  merged_cgraph_to_pynx.convert_merged_cost_graph();
  py::object nx_graph = merged_cgraph_to_pynx.get_py_graph().attr("get_nx_graph")();
  pickle_input.attr("set_nx_graph")(nx_graph);


  // 查找 args 类的某个参数
  py::object args_test = py::module::import("python.py_input") \
                           .attr("Args")(py_dict, reinforce_params);

  // 将 Args 里的参数以 txt 的形式输出
  args_test.attr("log_config")();
  args_test.attr("startup_strategy")(pickle_input);

}