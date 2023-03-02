#include <cstdio>
#include <string>
#include <map>
#include <iostream>

#include "aware_fusion.h"

int main() {
  Graph graph;
  NodeBase node_1;
  node_1.set_name("input");
  node_1.set_device("dev0");
  node_1.set_compute_cost(2.0);
  node_1.set_inputs({});
  node_1.set_input_comm_costs({});
  node_1.set_persistent_memory(1.5);
  node_1.set_outputs({"conv1", "conv2"});
  node_1.set_output_comm_costs({2, 4});

  NodeBase node_2;
  node_2.set_name("conv1");
  node_2.set_device("dev1");
  node_2.set_compute_cost(3.0);
  node_2.set_inputs({"input"});
  node_2.set_input_comm_costs({2});
  node_2.set_persistent_memory(3.6);
  node_2.set_outputs({"conv2"});
  node_2.set_output_comm_costs({3});

  NodeBase node_3;
  node_3.set_name("conv2");
  node_3.set_device("dev2");
  node_3.set_compute_cost(3.3);
  node_3.set_inputs({"input", "conv1"});
  node_3.set_input_comm_costs({4, 3});
  node_3.set_persistent_memory(3.4);
  node_3.set_outputs({"conv3"});
  node_3.set_output_comm_costs({3});

  NodeBase node_4;
  node_4.set_name("conv3");
  node_4.set_device("dev0");
  node_4.set_compute_cost(3.5);
  node_4.set_inputs({"conv2"});
  node_4.set_input_comm_costs({3});
  node_4.set_persistent_memory(3.9);
  node_4.set_outputs({"relu"});
  node_4.set_output_comm_costs({3});

  NodeBase node_5;
  node_5.set_name("relu");
  node_5.set_device("dev1");
  node_5.set_compute_cost(3.1);
  node_5.set_inputs({"conv3"});
  node_5.set_input_comm_costs({3});
  node_5.set_persistent_memory(2.4);
  node_5.set_outputs({"output"});
  node_5.set_output_comm_costs({3});

  NodeBase node_6;
  node_5.set_name("output");
  node_5.set_device("dev3");
  node_5.set_compute_cost(3.1);
  node_5.set_inputs({"relu"});
  node_5.set_input_comm_costs({3});
  node_5.set_persistent_memory(2.6);
  node_5.set_outputs({});
  node_5.set_output_comm_costs({});

  std::vector<NodeBase> nodes = {node_1, node_2, node_3, 
                                 node_4, node_5, node_6};

  graph.set_nodes(nodes);

  std::map<std::string, NodeBase&> node_map;
  node_map.insert(std::pair<std::string, NodeBase&>(node_1.get_name(), node_1));
  node_map.insert(std::pair<std::string, NodeBase&>(node_2.get_name(), node_2));
  node_map.insert(std::pair<std::string, NodeBase&>(node_3.get_name(), node_3));
  node_map.insert(std::pair<std::string, NodeBase&>(node_4.get_name(), node_4));
  node_map.insert(std::pair<std::string, NodeBase&>(node_5.get_name(), node_5)); 
  node_map.insert(std::pair<std::string, NodeBase&>(node_6.get_name(), node_6));

  graph.set_node_map(node_map);
  int threshold = 4;

  AwareFusion aware_fusion(graph, threshold);
  aware_fusion.GenerateFusedGraph();
  std::vector<MergedCostNode&> merged_cost_node_vec = aware_fusion.get_merged_cost_node_vector();
  std::map<std::string, MergedCostNode&> merged_cost_node_map = aware_fusion.get_merged_cost_node_map();

  for (auto merged_cost_node : merged_cost_node_vec) {
    std::cout << "name: " << merged_cost_node.get_name() << std::endl;
    std::cout << "compute cost: " << merged_cost_node.get_compute_cost() << std::endl;
    std::cout << "memory cost: " << merged_cost_node.get_memory_cost() << std::endl;
    for (auto cost_node : merged_cost_node.get_megred_cost_nodes()) {
      std::cout << "cost node name: " << cost_node.get_name() << std::endl;
    }
    for (int i = 0; i < merged_cost_node.get_inputs().size(); i ++) {
      std::cout << "input: " << merged_cost_node.get_inputs()[i] << 
                  "comm cost: " << merged_cost_node.get_input_comm_costs()[i] << std::endl; 
    }
    for (int i = 0; i < merged_cost_node.get_outputs().size(); i ++) {
      std::cout << "output: " << merged_cost_node.get_outputs()[i] << 
                  "comm cost: " << merged_cost_node.get_output_comm_costs()[i] << std::endl; 
    }
  }

  for (auto merged_cost_node : merged_cost_node_map) {
    std::cout << "name: " << merged_cost_node.second.get_name() << std::endl;
    std::cout << "compute cost: " << merged_cost_node.second.get_compute_cost() << std::endl;
    std::cout << "memory cost: " << merged_cost_node.second.get_memory_cost() << std::endl;
    for (auto cost_node : merged_cost_node.second.get_megred_cost_nodes()) {
      std::cout << "cost node name: " << cost_node.get_name() << std::endl;
    }
    for (int i = 0; i < merged_cost_node.second.get_inputs().size(); i ++) {
      std::cout << "input: " << merged_cost_node.second.get_inputs()[i] << 
                  "comm cost: " << merged_cost_node.second.get_input_comm_costs()[i] << std::endl; 
    }
    for (int i = 0; i < merged_cost_node.second.get_outputs().size(); i ++) {
      std::cout << "output: " << merged_cost_node.second.get_outputs()[i] << 
                  "comm cost: " << merged_cost_node.second.get_output_comm_costs()[i] << std::endl; 
    }
  }

}