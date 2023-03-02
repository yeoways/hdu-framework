#include <pynx_to_cgraph.h>
#include <string>
#include <vector>

void ConvertPYNXToCGraph::convert_py_nx_graph(){
  std::vector<CostNode> cost_nodes;
  for (auto node : nx_graph.attr("nodes")()) {
    CostNode cost_node;
    cost_node.set_name(node.cast<std::string>());
    cost_node.set_compute_cost(nx_graph.attr("nodes")[node]["cost"].cast<long>());
    cost_node.set_memory_cost(nx_graph.attr("nodes")[node]["mem"].cast<long>());
    std::vector<std::string> outputs;
    py::object out_edges = nx_graph.attr("out_edges")(node);
    py::object util = py::module::import("python.util");
    int node_num = util.attr("get_len")(out_edges).cast<int>();
    for (auto out_edge : out_edges) {
      // output 是一个二元组 (in_node, out_node)
      std::string out = util.attr("get_tuple_element")(out_edge, 1).cast<std::string>();
      outputs.push_back(out);
    }
    cost_node.set_outputs(outputs);
    std::vector<long> comm_costs(node_num, nx_graph.attr("nodes")[node]["out_size"].cast<long>() / node_num);
    cost_node.set_output_comm_costs(comm_costs);
    cost_node.set_device(nx_graph.attr("nodes")[node]["device"].cast<std::string>());
    cost_nodes.push_back(cost_node);
  }
  cost_graph.set_cost_nodes(cost_nodes);
}

void ConvertPYNXToCMergedGraph::convert_py_nx_graph() {
  std::vector<MergedCostNode> merged_cost_nodes;
  for (auto node : nx_graph.attr("nodes")()) {
    MergedCostNode merged_cost_node;
    merged_cost_node.set_name(node.cast<std::string>());
    merged_cost_node.set_compute_cost(nx_graph.attr("nodes")[node]["cost"].cast<long>());
    merged_cost_node.set_memory_cost(nx_graph.attr("nodes")[node]["mem"].cast<long>());
    std::vector<std::string> outputs;
    py::object out_edges = nx_graph.attr("out_edges")(node);
    py::object util = py::module::import("python.util");
    int node_num = util.attr("get_len")(out_edges).cast<int>();
    for (auto out_edge : out_edges) {
      // output 是一个二元组 (in_node, out_node)
      std::string out = util.attr("get_tuple_element")(out_edge, 1).cast<std::string>();
      outputs.push_back(out);
    }
    merged_cost_node.set_outputs(outputs);
    std::vector<long> comm_costs(node_num, nx_graph.attr("nodes")[node]["out_size"].cast<long>() / node_num);
    merged_cost_node.set_output_comm_costs(comm_costs);
    merged_cost_node.set_device(nx_graph.attr("nodes")[node]["device"].cast<std::string>());
    std::vector<std::string> merged_cost_node_names;
    py::object aggregated_nodes = nx_graph.attr("nodes")[node]["aggregated_nodes"];
    for (auto aggregated_node : aggregated_nodes) {
      std::string merged_cost_node_name = aggregated_node.cast<std::string>();
      merged_cost_node_names.push_back(merged_cost_node_name);
    }
    merged_cost_node.set_merged_cost_node_names(merged_cost_node_names);
    merged_cost_nodes.push_back(merged_cost_node);
  }
  merged_cost_graph.set_merged_cost_nodes(merged_cost_nodes);  
}