#include "aware_fusion.h"

#define upper_limit 4

AwareFusion::AwareFusion(Graph graph, int threshold): graph(graph),
                                                      threshold(threshold) 
{
  cost_graph = ConvertGraphToCostGraph(graph);
}

void AwareFusion::GenerateFusedGraph() {
  std::map<std::string, MergedCostNode&> merged_cost_nodes;
  int node_num = cost_graph.get_cost_nodes().size();
  std::vector<MergedCostNode&> merged_cost_node_vec;
  std::vector<CostNode> cost_nodes = cost_graph.get_cost_nodes();
  for (int i = 0; i < node_num; i ++) {
    CostNode cost_node = cost_nodes[i];
    MergedCostNode merged_cost_node(cost_node);
    merged_cost_node_vec.push_back(merged_cost_node);
    merged_cost_nodes.insert(
      std::pair<std::string, MergedCostNode&>(merged_cost_node.get_name(), merged_cost_node));
    
  }

  int cycle_time = 0;
  while (merged_cost_node_vec.size() > threshold || cycle_time <= upper_limit) {
    OperatorFusing(merged_cost_node_vec, merged_cost_nodes);
    cycle_time += 1;
  }

  // 最后生成的是一个 merged_cost_node_vec, merged_cost_nodes
  set_merged_cost_node_vector(merged_cost_node_vec);
  set_merged_cost_node_map(merged_cost_nodes);

}

void AwareFusion::OperatorFusing(std::vector<MergedCostNode&>& merged_cost_node_vec, 
  std::map<std::string, MergedCostNode&>& merged_cost_nodes) {

  for (MergedCostNode& merged_cost_node : merged_cost_node_vec) {
    std::vector<std::string>& inputs = merged_cost_node.get_inputs();
    std::vector<std::string>& outputs = merged_cost_node.get_outputs();
    std::vector<long>& input_comm_costs = merged_cost_node.get_input_comm_costs();
    std::vector<long>& output_comm_costs = merged_cost_node.get_output_comm_costs();
    // 论文第一个判断条件, 出度必须为 1 才有资格合并
    if (outputs.size() == 1) {
      // 然后追溯后继节点, 准备第二个条件
      MergedCostNode& succ_node = merged_cost_nodes.find(outputs[0])->second;
      long total_compute_cost = merged_cost_node.get_compute_cost() + succ_node.get_compute_cost();
      long comm_cost = input_comm_costs[0];
      // 论文的第二个条件, 两个节点的通信开销要大于两个节点计算开销的均值
      if (comm_cost >= total_compute_cost / 2) {
        MergedCostNode node(merged_cost_node, succ_node);
        // 对原先地址上的原节点替换成合并后的节点, 后继节点从vector和map上删除
        merged_cost_node = node;
        for (int i = 0; i < merged_cost_node_vec.size(); i ++) {
          if (merged_cost_node_vec[i].get_name() == succ_node.get_name()) {
            merged_cost_node_vec.erase(merged_cost_node_vec.begin() + i);
          }
        }
        merged_cost_nodes.erase(outputs[0]);
        // 合并后的节点包含之前合并了两个节点的inputs和outputs, 
        // 此时必须要更正inputs和outputs中的节点中所指向的合并后的节点名
        UpdateNodeInfo(merged_cost_node_vec, merged_cost_nodes, node);

      }
    }
  }
}

void AwareFusion::UpdateNodeInfo (std::vector<MergedCostNode&>& merged_cost_node_vec, 
  std::map<std::string, MergedCostNode&>& merged_cost_nodes, MergedCostNode& node) {
  
  std::vector<std::string> inputs = node.get_inputs();
  std::vector<std::string> outputs = node.get_outputs();
  std::vector<CostNode> cost_nodes = node.get_megred_cost_nodes();
  std::vector<std::string> cost_node_names;

  // 获取融合节点合并的各个节点名称
  for (CostNode& cost_node : cost_nodes) {
    cost_node_names.push_back(cost_node.get_name());
  }
  
  // 修改节点的信息
  // 前向节点
  int loc;
  for (std::string input : inputs) {
    for (MergedCostNode& merged_cost_node : merged_cost_node_vec) {
      if (merged_cost_node.get_name() == input) {
        std::vector<std::string>& outputs = merged_cost_node.get_outputs();
        std::vector<long>& output_comm_costs = merged_cost_node.get_output_comm_costs();
        // 解释: 例如有一个前向节点同时指向了融合前的两个节点, 
        // 因此该前向节点的outputs中的对应两个output字段需要合并, 
        // 采取的方案是将遇到的第一个字段名改成融合之后的节点名
        // 将第二个字段名删除, 并且记得将其对应的通信代价加到第一个字段名对应的代价中
        for (int i = 0; i < outputs.size(); i ++) {
          int first_time = 1;
          for (CostNode& cost_node : cost_nodes) {
            if (outputs[i] == cost_node.get_name()) {
              // 如果是第一个, 直接改名
              if (first_time == 1) {
                outputs[i] = node.get_name();
                first_time = 0;
                loc = i;
              }
              // 如果这个节点是融合
              else {
                output_comm_costs[loc] += output_comm_costs[i];
                outputs.erase(outputs.begin() + i);
                output_comm_costs.erase(output_comm_costs.begin() + i);
              }
            }
          }
        }
      }
    }
  }

  // 后继节点
  int loc;
  for (std::string output : outputs) {
    for (MergedCostNode& merged_cost_node : merged_cost_node_vec) {
      if (merged_cost_node.get_name() == output) {
        std::vector<std::string>& inputs = merged_cost_node.get_outputs();
        std::vector<long>& input_comm_costs = merged_cost_node.get_output_comm_costs();
        for (int i = 0; i < inputs.size(); i ++) {
          int first_time = 1;
          for (CostNode& cost_node : cost_nodes) {
            if (inputs[i] == cost_node.get_name()) {
              // 如果是第一个, 直接改名
              if (first_time == 1) {
                inputs[i] = node.get_name();
                first_time = 0;
                loc = i;
              }
              // 如果这个节点是融合
              else {
                input_comm_costs[loc] += input_comm_costs[i];
                inputs.erase(inputs.begin() + i);
                input_comm_costs.erase(input_comm_costs.begin() + i);
              }
            }
          }
        }
      }
    }
  }


}

