#include <cgraph_to_pynx.h>
#include <pynx_to_cgraph.h>
#include <iostream>
#include <map>
using namespace std;
int main(){
    py::scoped_interpreter python; //一定要加, 否则会报错

    py::dict py_dict;
    py_dict["pickled_inp_file"] = "/mnt/e/DatasetWareHouse/datasets/ptb/100843/input.pkl";
    py_dict["n_devs"] = 4;
    py_dict["num_cpus"] = 1;
    py_dict["start_virtual_GPU"] = true;
    py_dict["allow_soft_placement"] = true;
    py_dict["disable_detailed_stats"] = false;
    py_dict["disable_timeline"] = false;
    py_dict["verbose"] = true;
    py_dict["step"] = 5000;
    py_dict["issim"] = true;
    py_dict["isbase"] = false;
    py_dict["isall"] = true;
    py_dict["output_dir"] = ../output/nmt_Trinity/4/;
    py::object pickle_input = py::module::import("python.py_input") \
                           .attr("Pickle")(py_dict["pickled_inp_file"]);
    py::object op_pref = pickle_input.attr("get_op_pref")();
    py::object step_stats = pickle_input.attr("get_step_stats")();
    
    ConvertPYNXToCMergedGraph pynx_to_merged_cgraph(pickle_input.attr("get_nx_graph")());
    pynx_to_merged_cgraph.convert_py_nx_graph();
    MergedCostGraph merged_cost_graph = pynx_to_merged_cgraph.get_merged_cost_graph();

    // 算子融合部分还没接上, 这部分补上整个流程就打通了

    ConvertCMergedGraphToPYNX merged_cgraph_to_pynx(merged_cost_graph);
    // 转换成nx_graph.py中的graph形式
    merged_cgraph_to_pynx.convert_merged_cost_graph();
    py::object nx_graph = merged_cgraph_to_pynx.get_py_graph().attr("get_nx_graph")();
    pickle_input.attr("set_nx_graph")(nx_graph);

    py:object hparams = py::module::import("python.Trinity_program_norml") \
                            .attr("trinity_mian_hparams")();
    py:object gcontroller = py::module::import("cluster") \
                            .attr("TrinityControllerTest")(py_dict["n_devs"],py_dict["num_cpus"],py_dict["start_virtual_GPU"] \
                            , py_dict["allow_soft_placement"] , py_dict["disable_detailed_stats"], py_dict["disable_timeline"])
    py:object gcluster = gcontroller.attr("getCluster")()
    py::object graph_placer = py::module::import("python.graph_scheduling") \
                            .attr("GraphScheduling")(nx_graph,gcluster,op_pref,step_stats,hparams,py_dict["verbose"],py_dict["step"],py_dict["issim"],py_dict["isbase"]);
    cout << "-----------------开始调度---------------" << endl;
    py::object placed_mg = graph_placer.attr("schedule_graph")(py_dict["isall"],py_dict["issim"],py_dict["output_dir"],py_dict["isbase"]);
    cout << "------------------使用Trinity方法搜索最优模型并行策略结束----------------------" << endl;
}
