#include <cgraph_to_pynx.h>
#include <numeric>

void ConvertCGraphToPYNX::convert_cost_graph(){
  for (auto node : cost_graph.get_cost_nodes()) {
    std::string name = node.get_name();
    long cost = node.get_compute_cost();
    long mem = node.get_memory_cost();
    std::vector<long> out_sizes = node.get_output_comm_costs();
    long out_size = accumulate(out_sizes.begin(), out_sizes.end(), 0);
    std::string device = node.get_device();
    std::vector<std::string> outputs = node.get_outputs();
    nx_graph_add_node(name, cost, mem, out_size, device);
    for (std::string output : outputs) {
      nx_graph_add_edge(name, output);
    }
  }
}

void ConvertCGraphToPYNX::nx_graph_add_node(std::string name, 
                                            long cost,
                                            long mem,
                                            long out_size,
                                            std::string device){
  py_graph.attr("add_node")(name, cost, mem, out_size, device);
}

void ConvertCGraphToPYNX::nx_graph_add_edge(std::string from,
                                            std::string to){
  py_graph.attr("add_edge")(from, to);                                        
}

int ConvertCGraphToPYNX::number_of_nodes(){
  return py_graph.attr("number_of_nodes")().cast<int>();
}

void ConvertCMergedGraphToPYNX::convert_merged_cost_graph(){
  for (auto node : merged_cost_graph.get_merged_cost_nodes()) {
    std::string name = node.get_name();
    long cost = node.get_compute_cost();
    long mem = node.get_memory_cost();
    std::vector<long> out_sizes = node.get_output_comm_costs();
    long out_size = accumulate(out_sizes.begin(), out_sizes.end(), 0);
    std::string device = node.get_device();
    std::vector<std::string> outputs = node.get_outputs();
    nx_graph_add_node(name, cost, mem, out_size, device);
    for (std::string output : outputs) {
      nx_graph_add_edge(name, output);
    }
  }
};