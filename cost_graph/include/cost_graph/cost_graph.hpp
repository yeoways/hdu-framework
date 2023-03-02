#include <vector>
#include <string>
#include <map>
#include <algorithm>

#include <DistributedIR/node.hpp>
#include <DistributedIR/graph.hpp>
#include "util.hpp"

#ifndef _FRAMEWORK_COST_GRAPH_COST_GRAPH_H
#define _FRAMEWORK_COST_GRAPH_COST_GRAPH_H

namespace framework {

class CostNode {
  private:
    std::string name;
    std::string device;
    long compute_cost;
    long memory_cost;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<long> input_comm_costs;
    std::vector<long> output_comm_costs;
  
  public:
    CostNode() {}
    CostNode(NodeBase node)
      : name(node.get_name()),
        device(node.get_device()),
        compute_cost(node.get_compute_cost()),
        memory_cost(node.get_persistent_memory()),
        inputs(node.get_inputs()),
        outputs(node.get_outputs()),
        input_comm_costs(node.get_input_comm_costs()),
        output_comm_costs(node.get_output_comm_costs()) {}       
    virtual ~CostNode() {}
    GEN_ACCESSOR_IN_DEC(std::string, name)
    GEN_ACCESSOR_IN_DEC(std::string, device)
    GEN_ACCESSOR_IN_DEC(long, compute_cost)
    GEN_ACCESSOR_IN_DEC(long, memory_cost)
    GEN_ACCESSOR_IN_DEC(std::vector<std::string>, inputs)
    GEN_ACCESSOR_IN_DEC(std::vector<std::string>, outputs)
    GEN_ACCESSOR_IN_DEC(std::vector<long>, input_comm_costs)
    GEN_ACCESSOR_IN_DEC(std::vector<long>, output_comm_costs)
};

class Edge {
  private:
    std::string right_node;
    std::string left_node;
    long comm_cost;
  
  public:
    Edge() {}
    Edge(Edge* edge)
      : right_node(edge->get_right_node()),
        left_node(edge->get_left_node()),
        comm_cost(edge->get_comm_cost()) {}
    virtual ~Edge() {}
    GEN_ACCESSOR_IN_DEC(std::string, right_node)
    GEN_ACCESSOR_IN_DEC(std::string, left_node)
    GEN_ACCESSOR_IN_DEC(long, comm_cost)
};

class CostGraph {
  private:
    std::vector<CostNode> cost_nodes;
    std::map<std::string, CostNode&> cost_node_map;
    std::map<std::string, std::vector<Edge>> succ_edges;
    std::map<std::string, std::vector<Edge>> pred_edges;

  public:
    CostGraph() {}
    CostGraph(CostGraph* cost_graph)
      : cost_nodes(cost_graph->get_cost_nodes()),
        cost_node_map(cost_graph->get_cost_node_map()),
        succ_edges(cost_graph->get_succ_edges()),
        pred_edges(cost_graph->get_pred_edges()) {}
    virtual ~CostGraph() {}
    GEN_ACCESSOR_IN_DEC(std::vector<CostNode>, cost_nodes)
    GEN_ACCESSOR_IN_DEC(ALL(std::map<std::string, CostNode&>), cost_node_map)
    GEN_ACCESSOR_IN_DEC(ALL(std::map<std::string, std::vector<Edge>>), succ_edges)
    GEN_ACCESSOR_IN_DEC(ALL(std::map<std::string, std::vector<Edge>>), pred_edges)
    void add_node(CostNode node) {
      cost_node_map.insert(std::pair<std::string, CostNode&>(node.get_name(), node));
      cost_nodes.push_back(node);
    }
    CostNode& get_node(std::string name) { return cost_node_map.find(name)->second; }

};

// 这是一个消融的cost node的定义, 之后可能要做修改, 暂定这个格式
class MergedCostNode : public CostNode{
  private:
    std::vector<CostNode> megred_cost_nodes;
    std::vector<std::string> merged_cost_node_names;

  public:
    MergedCostNode () {}
    // 合并两个 cost node
    MergedCostNode (CostNode node_1, CostNode node_2) {
      set_name(node_1.get_name() + node_2.get_name());
      set_device(node_1.get_device());
      set_compute_cost(node_1.get_compute_cost() + node_2.get_compute_cost());
      set_memory_cost(node_1.get_memory_cost() + node_2.get_memory_cost());

      // 当两个节点融合时, 它们边的代价也要融合
      std::vector<std::string> inputs_1 = node_1.get_inputs();
      std::vector<std::string> inputs_2 = node_2.get_inputs();
      std::vector<long> input_comm_costs_1 = node_1.get_input_comm_costs();
      std::vector<long> input_comm_costs_2 = node_2.get_input_comm_costs();
      std::map<std::string, long> input_to_comm_costs_1;
      std::map<std::string, long> input_to_comm_costs_2;
      std::vector<std::string> inputs;
      std::vector<long> input_comm_costs;
      for (int i = 0; i < inputs_1.size(); i ++) {
        input_to_comm_costs_1.insert(std::pair<std::string, long>(inputs_1[i], input_comm_costs_1[i]));
      }
      for (int i = 0; i < inputs_2.size(); i ++) {
        input_to_comm_costs_2.insert(std::pair<std::string, long>(inputs_2[i], input_comm_costs_2[i]));
      }

      for (int i = 0; i < inputs_1.size(); i ++) {
        std::string name = inputs[i];
        if (input_to_comm_costs_2.find(name) != input_to_comm_costs_2.end()) {
          long cost_1 = input_to_comm_costs_1[name];
          long cost_2 = input_to_comm_costs_2[name];
          inputs.push_back(name);
          input_comm_costs.push_back(cost_1 + cost_2);
          input_to_comm_costs_2.erase(name);
        }
        else {
          long cost_1 = input_to_comm_costs_1[name];
          inputs.push_back(name);
          input_comm_costs.push_back(cost_1);
        }
      }
      for (auto input_to_comm_cost_2 : input_to_comm_costs_2) {
        std::string name = input_to_comm_cost_2.first;
        long cost_2 = input_to_comm_cost_2.second;
        // 如果这条边是 node_1 和 node_2 之间的边, 直接忽略, 融合之后的算子间理论上不应该存在边
        if (name == node_1.get_name()) {
          inputs.push_back(name);
          input_comm_costs.push_back(0);
          continue;
        }
        inputs.push_back(name);
        input_comm_costs.push_back(cost_2);
      }
      set_inputs(inputs);
      set_input_comm_costs(input_comm_costs);

      std::vector<std::string> outputs_1 = node_1.get_outputs();
      std::vector<std::string> outputs_2 = node_2.get_outputs();
      std::vector<long> output_comm_costs_1 = node_1.get_output_comm_costs();
      std::vector<long> output_comm_costs_2 = node_2.get_output_comm_costs();
      std::map<std::string, long> output_to_comm_costs_1;
      std::map<std::string, long> output_to_comm_costs_2;
      std::vector<std::string> outputs;
      std::vector<long> output_comm_costs;
      for (int i = 0; i < outputs_1.size(); i ++) {
        output_to_comm_costs_1.insert(std::pair<std::string, long>(outputs_1[i], output_comm_costs_1[i]));
      }
      for (int i = 0; i < outputs_2.size(); i ++) {
        output_to_comm_costs_2.insert(std::pair<std::string, long>(outputs_2[i], output_comm_costs_2[i]));
      }

      for (int i = 0; i < outputs_2.size(); i ++) {
        std::string name = outputs[i];
        if (output_to_comm_costs_1.find(name) != output_to_comm_costs_1.end()) {
          long cost_1 = output_to_comm_costs_1[name];
          long cost_2 = output_to_comm_costs_2[name];
          outputs.push_back(name);
          output_comm_costs.push_back(cost_1 + cost_2);
          output_to_comm_costs_1.erase(name);
        }
        else {
          long cost_2 = output_to_comm_costs_2[name];
          outputs.push_back(name);
          output_comm_costs.push_back(cost_2);
        }
      }
      for (auto output_to_comm_cost_1 : output_to_comm_costs_1) {
        std::string name = output_to_comm_cost_1.first;
        long cost_1 = output_to_comm_cost_1.second;
        // 如果这条边是 node_1 和 node_2 之间的边, 直接忽略, 融合之后的算子间理论上不应该存在边
        if (name == node_2.get_name()) {
          inputs.push_back(name);
          input_comm_costs.push_back(0);
          continue;
        }
        outputs.push_back(name);
        output_comm_costs.push_back(cost_1);
      }
      set_outputs(outputs);
      set_output_comm_costs(output_comm_costs);

      set_megred_cost_nodes({node_1, node_2});
    }
    // 将单个cost node转换成一个merged node, 一般用来初始化
    MergedCostNode (CostNode node) {
      set_name(node.get_name());
      set_device(node.get_device());
      set_compute_cost(node.get_compute_cost());
      set_memory_cost(node.get_memory_cost());
      set_inputs(node.get_inputs());
      set_input_comm_costs(node.get_input_comm_costs());
      set_outputs(node.get_outputs());
      set_output_comm_costs(node.get_output_comm_costs());
      set_megred_cost_nodes({node});
    }
    // 合并两个Merged node
    MergedCostNode (MergedCostNode node_1, MergedCostNode node_2) {
      set_name(node_1.get_name() + node_2.get_name()); // 这个字段暂时没用
      set_device(node_1.get_device());
      set_compute_cost(node_1.get_compute_cost() + node_2.get_compute_cost());
      set_memory_cost(node_1.get_memory_cost() + node_2.get_memory_cost());

      // 当两个节点融合时, 它们边的代价也要融合
      std::vector<std::string> inputs_1 = node_1.get_inputs();
      std::vector<std::string> inputs_2 = node_2.get_inputs();
      std::vector<long> input_comm_costs_1 = node_1.get_input_comm_costs();
      std::vector<long> input_comm_costs_2 = node_2.get_input_comm_costs();
      std::map<std::string, long> input_to_comm_costs_1;
      std::map<std::string, long> input_to_comm_costs_2;
      std::vector<std::string> inputs;
      std::vector<long> input_comm_costs;
      for (int i = 0; i < inputs_1.size(); i ++) {
        input_to_comm_costs_1.insert(std::pair<std::string, long>(inputs_1[i], input_comm_costs_1[i]));
      }
      for (int i = 0; i < inputs_2.size(); i ++) {
        input_to_comm_costs_2.insert(std::pair<std::string, long>(inputs_2[i], input_comm_costs_2[i]));
      }

      for (int i = 0; i < inputs_1.size(); i ++) {
        std::string name = inputs[i];
        if (input_to_comm_costs_2.find(name) != input_to_comm_costs_2.end()) {
          long cost_1 = input_to_comm_costs_1[name];
          long cost_2 = input_to_comm_costs_2[name];
          inputs.push_back(name);
          input_comm_costs.push_back(cost_1 + cost_2);
          input_to_comm_costs_2.erase(name);
        }
        else {
          long cost_1 = input_to_comm_costs_1[name];
          inputs.push_back(name);
          input_comm_costs.push_back(cost_1);
        }
      }
      for (auto input_to_comm_cost_2 : input_to_comm_costs_2) {
        std::string name = input_to_comm_cost_2.first;
        long cost_2 = input_to_comm_cost_2.second;
        // 如果这条边是 node_1 和 node_2 之间的边, 设置为0
        if (name == node_1.get_name()) {
          inputs.push_back(name);
          input_comm_costs.push_back(0);
          continue;
        }
        inputs.push_back(name);
        input_comm_costs.push_back(cost_2);
      }
      set_inputs(inputs);
      set_input_comm_costs(input_comm_costs);

      std::vector<std::string> outputs_1 = node_1.get_outputs();
      std::vector<std::string> outputs_2 = node_2.get_outputs();
      std::vector<long> output_comm_costs_1 = node_1.get_output_comm_costs();
      std::vector<long> output_comm_costs_2 = node_2.get_output_comm_costs();
      std::map<std::string, long> output_to_comm_costs_1;
      std::map<std::string, long> output_to_comm_costs_2;
      std::vector<std::string> outputs;
      std::vector<long> output_comm_costs;
      for (int i = 0; i < outputs_1.size(); i ++) {
        output_to_comm_costs_1.insert(std::pair<std::string, long>(outputs_1[i], output_comm_costs_1[i]));
      }
      for (int i = 0; i < outputs_2.size(); i ++) {
        output_to_comm_costs_2.insert(std::pair<std::string, long>(outputs_2[i], output_comm_costs_2[i]));
      }

      for (int i = 0; i < outputs_2.size(); i ++) {
        std::string name = outputs[i];
        if (output_to_comm_costs_1.find(name) != output_to_comm_costs_1.end()) {
          long cost_1 = output_to_comm_costs_1[name];
          long cost_2 = output_to_comm_costs_2[name];
          outputs.push_back(name);
          output_comm_costs.push_back(cost_1 + cost_2);
          output_to_comm_costs_1.erase(name);
        }
        else {
          long cost_2 = output_to_comm_costs_2[name];
          outputs.push_back(name);
          output_comm_costs.push_back(cost_2);
        }
      }
      for (auto output_to_comm_cost_1 : output_to_comm_costs_1) {
        std::string name = output_to_comm_cost_1.first;
        long cost_1 = output_to_comm_cost_1.second;
        // 如果这条边是 node_1 和 node_2 之间的边, 直接忽略, 融合之后的算子间理论上不应该存在边
        if (name == node_2.get_name()) {
          inputs.push_back(name);
          input_comm_costs.push_back(0);
          continue;
        }
        outputs.push_back(name);
        output_comm_costs.push_back(cost_1);
      }
      set_outputs(outputs);
      set_output_comm_costs(output_comm_costs);

      std::vector<CostNode> cost_nodes_1 = node_1.get_megred_cost_nodes();
      std::vector<CostNode> cost_nodes_2 = node_2.get_megred_cost_nodes();
      cost_nodes_1.insert(cost_nodes_1.end(), cost_nodes_2.begin(), cost_nodes_2.end());
      set_megred_cost_nodes(cost_nodes_1);      
    }

    MergedCostNode (MergedCostNode node_1, CostNode node_2) {
      set_name(node_1.get_name() + node_2.get_name()); // 这个字段暂时没用
      set_device(node_1.get_device());
      set_compute_cost(node_1.get_compute_cost() + node_2.get_compute_cost());
      set_memory_cost(node_1.get_memory_cost() + node_2.get_memory_cost());

      // 当两个节点融合时, 它们边的代价也要融合
      std::vector<std::string> inputs_1 = node_1.get_inputs();
      std::vector<std::string> inputs_2 = node_2.get_inputs();
      std::vector<long> input_comm_costs_1 = node_1.get_input_comm_costs();
      std::vector<long> input_comm_costs_2 = node_2.get_input_comm_costs();
      std::map<std::string, long> input_to_comm_costs_1;
      std::map<std::string, long> input_to_comm_costs_2;
      std::vector<std::string> inputs;
      std::vector<long> input_comm_costs;
      for (int i = 0; i < inputs_1.size(); i ++) {
        input_to_comm_costs_1.insert(std::pair<std::string, long>(inputs_1[i], input_comm_costs_1[i]));
      }
      for (int i = 0; i < inputs_2.size(); i ++) {
        input_to_comm_costs_2.insert(std::pair<std::string, long>(inputs_2[i], input_comm_costs_2[i]));
      }

      for (int i = 0; i < inputs_1.size(); i ++) {
        std::string name = inputs[i];
        if (input_to_comm_costs_2.find(name) != input_to_comm_costs_2.end()) {
          long cost_1 = input_to_comm_costs_1[name];
          long cost_2 = input_to_comm_costs_2[name];
          inputs.push_back(name);
          input_comm_costs.push_back(cost_1 + cost_2);
          input_to_comm_costs_2.erase(name);
        }
        else {
          long cost_1 = input_to_comm_costs_1[name];
          inputs.push_back(name);
          input_comm_costs.push_back(cost_1);
        }
      }
      for (auto input_to_comm_cost_2 : input_to_comm_costs_2) {
        std::string name = input_to_comm_cost_2.first;
        long cost_2 = input_to_comm_cost_2.second;
        // 如果这条边是 node_1 和 node_2 之间的边, 设置为0
        if (name == node_1.get_name()) {
          inputs.push_back(name);
          input_comm_costs.push_back(0);
          continue;
        }
        inputs.push_back(name);
        input_comm_costs.push_back(cost_2);
      }
      set_inputs(inputs);
      set_input_comm_costs(input_comm_costs);

      std::vector<std::string> outputs_1 = node_1.get_outputs();
      std::vector<std::string> outputs_2 = node_2.get_outputs();
      std::vector<long> output_comm_costs_1 = node_1.get_output_comm_costs();
      std::vector<long> output_comm_costs_2 = node_2.get_output_comm_costs();
      std::map<std::string, long> output_to_comm_costs_1;
      std::map<std::string, long> output_to_comm_costs_2;
      std::vector<std::string> outputs;
      std::vector<long> output_comm_costs;
      for (int i = 0; i < outputs_1.size(); i ++) {
        output_to_comm_costs_1.insert(std::pair<std::string, long>(outputs_1[i], output_comm_costs_1[i]));
      }
      for (int i = 0; i < outputs_2.size(); i ++) {
        output_to_comm_costs_2.insert(std::pair<std::string, long>(outputs_2[i], output_comm_costs_2[i]));
      }

      for (int i = 0; i < outputs_2.size(); i ++) {
        std::string name = outputs[i];
        if (output_to_comm_costs_1.find(name) != output_to_comm_costs_1.end()) {
          long cost_1 = output_to_comm_costs_1[name];
          long cost_2 = output_to_comm_costs_2[name];
          outputs.push_back(name);
          output_comm_costs.push_back(cost_1 + cost_2);
          output_to_comm_costs_1.erase(name);
        }
        else {
          long cost_2 = output_to_comm_costs_2[name];
          outputs.push_back(name);
          output_comm_costs.push_back(cost_2);
        }
      }
      for (auto output_to_comm_cost_1 : output_to_comm_costs_1) {
        std::string name = output_to_comm_cost_1.first;
        long cost_1 = output_to_comm_cost_1.second;
        // 如果这条边是 node_1 和 node_2 之间的边, 直接忽略, 融合之后的算子间理论上不应该存在边
        if (name == node_2.get_name()) {
          inputs.push_back(name);
          input_comm_costs.push_back(0);
          continue;
        }
        outputs.push_back(name);
        output_comm_costs.push_back(cost_1);
      }
      set_outputs(outputs);
      set_output_comm_costs(output_comm_costs);

      std::vector<CostNode> cost_nodes_1 = node_1.get_megred_cost_nodes();
      cost_nodes_1.push_back(node_2);
      set_megred_cost_nodes(cost_nodes_1);      
    }

    MergedCostNode (CostNode node_1, MergedCostNode node_2) {
      set_name(node_1.get_name() + node_2.get_name()); // 这个字段暂时没用
      set_device(node_1.get_device());
      set_compute_cost(node_1.get_compute_cost() + node_2.get_compute_cost());
      set_memory_cost(node_1.get_memory_cost() + node_2.get_memory_cost());

      // 当两个节点融合时, 它们边的代价也要融合
      std::vector<std::string> inputs_1 = node_1.get_inputs();
      std::vector<std::string> inputs_2 = node_2.get_inputs();
      std::vector<long> input_comm_costs_1 = node_1.get_input_comm_costs();
      std::vector<long> input_comm_costs_2 = node_2.get_input_comm_costs();
      std::map<std::string, long> input_to_comm_costs_1;
      std::map<std::string, long> input_to_comm_costs_2;
      std::vector<std::string> inputs;
      std::vector<long> input_comm_costs;
      for (int i = 0; i < inputs_1.size(); i ++) {
        input_to_comm_costs_1.insert(std::pair<std::string, long>(inputs_1[i], input_comm_costs_1[i]));
      }
      for (int i = 0; i < inputs_2.size(); i ++) {
        input_to_comm_costs_2.insert(std::pair<std::string, long>(inputs_2[i], input_comm_costs_2[i]));
      }

      for (int i = 0; i < inputs_1.size(); i ++) {
        std::string name = inputs[i];
        if (input_to_comm_costs_2.find(name) != input_to_comm_costs_2.end()) {
          long cost_1 = input_to_comm_costs_1[name];
          long cost_2 = input_to_comm_costs_2[name];
          inputs.push_back(name);
          input_comm_costs.push_back(cost_1 + cost_2);
          input_to_comm_costs_2.erase(name);
        }
        else {
          long cost_1 = input_to_comm_costs_1[name];
          inputs.push_back(name);
          input_comm_costs.push_back(cost_1);
        }
      }
      for (auto input_to_comm_cost_2 : input_to_comm_costs_2) {
        std::string name = input_to_comm_cost_2.first;
        long cost_2 = input_to_comm_cost_2.second;
        // 如果这条边是 node_1 和 node_2 之间的边, 设置为0
        if (name == node_1.get_name()) {
          inputs.push_back(name);
          input_comm_costs.push_back(0);
          continue;
        }
        inputs.push_back(name);
        input_comm_costs.push_back(cost_2);
      }
      set_inputs(inputs);
      set_input_comm_costs(input_comm_costs);

      std::vector<std::string> outputs_1 = node_1.get_outputs();
      std::vector<std::string> outputs_2 = node_2.get_outputs();
      std::vector<long> output_comm_costs_1 = node_1.get_output_comm_costs();
      std::vector<long> output_comm_costs_2 = node_2.get_output_comm_costs();
      std::map<std::string, long> output_to_comm_costs_1;
      std::map<std::string, long> output_to_comm_costs_2;
      std::vector<std::string> outputs;
      std::vector<long> output_comm_costs;
      for (int i = 0; i < outputs_1.size(); i ++) {
        output_to_comm_costs_1.insert(std::pair<std::string, long>(outputs_1[i], output_comm_costs_1[i]));
      }
      for (int i = 0; i < outputs_2.size(); i ++) {
        output_to_comm_costs_2.insert(std::pair<std::string, long>(outputs_2[i], output_comm_costs_2[i]));
      }

      for (int i = 0; i < outputs_2.size(); i ++) {
        std::string name = outputs[i];
        if (output_to_comm_costs_1.find(name) != output_to_comm_costs_1.end()) {
          long cost_1 = output_to_comm_costs_1[name];
          long cost_2 = output_to_comm_costs_2[name];
          outputs.push_back(name);
          output_comm_costs.push_back(cost_1 + cost_2);
          output_to_comm_costs_1.erase(name);
        }
        else {
          long cost_2 = output_to_comm_costs_2[name];
          outputs.push_back(name);
          output_comm_costs.push_back(cost_2);
        }
      }
      for (auto output_to_comm_cost_1 : output_to_comm_costs_1) {
        std::string name = output_to_comm_cost_1.first;
        long cost_1 = output_to_comm_cost_1.second;
        // 如果这条边是 node_1 和 node_2 之间的边, 直接忽略, 融合之后的算子间理论上不应该存在边
        if (name == node_2.get_name()) {
          inputs.push_back(name);
          input_comm_costs.push_back(0);
          continue;
        }
        outputs.push_back(name);
        output_comm_costs.push_back(cost_1);
      }
      set_outputs(outputs);
      set_output_comm_costs(output_comm_costs);

      std::vector<CostNode> cost_node_2 = node_2.get_megred_cost_nodes();
      cost_node_2.emplace(cost_node_2.begin(), node_1);
      set_megred_cost_nodes(cost_node_2);      
    }
    
    GEN_ACCESSOR_IN_DEC(std::vector<CostNode>, megred_cost_nodes)
    GEN_ACCESSOR_IN_DEC(std::vector<std::string>, merged_cost_node_names)
};

inline CostGraph ConvertGraphToCostGraph(Graph graph) {
  std::vector<NodeBase> graph_nodes = graph.get_nodes();
  int node_num = graph_nodes.size();
  std::vector<CostNode> cost_nodes;
  CostGraph cost_graph;
  std::map<std::string, std::vector<Edge>> succ_edges;
  std::map<std::string, std::vector<Edge>> pred_edges;

  for (int i = 0; i < node_num; i ++) {
    CostNode cost_node(graph_nodes[i]);

    std::string node_name = graph_nodes[i].get_name();
    std::vector<std::string> inputs = graph_nodes[i].get_inputs();
    std::vector<std::string> outputs = graph_nodes[i].get_outputs();
    std::vector<long> input_comm_costs = graph_nodes[i].get_input_comm_costs();
    std::vector<long> output_comm_costs = graph_nodes[i].get_output_comm_costs();
    int inputs_num = inputs.size();
    int outputs_num = outputs.size();
    std::vector<Edge> succ_edges_vec;
    std::vector<Edge> pred_edges_vec;
    
    for (int j = 0; j < inputs_num; j ++) {
      Edge edge;
      edge.set_left_node(inputs[j]);
      edge.set_right_node(node_name);
      edge.set_comm_cost(input_comm_costs[j]);
      pred_edges_vec.push_back(edge);
    }
    pred_edges.insert(
      std::pair<std::string, std::vector<Edge>>(node_name, pred_edges_vec));

    for (int j = 0; j < outputs_num; j ++) {
      Edge edge;
      edge.set_left_node(node_name);
      edge.set_right_node(outputs[j]);
      edge.set_comm_cost(output_comm_costs[j]);
      succ_edges_vec.push_back(edge);
    }
    succ_edges.insert(
      std::pair<std::string, std::vector<Edge>>(node_name, succ_edges_vec));    

    cost_graph.add_node(cost_node);
  }

  cost_graph.set_succ_edges(succ_edges);
  cost_graph.set_pred_edges(pred_edges);
  
  return cost_graph;

}

class MergedCostGraph {
  private:
    std::vector<MergedCostNode> merged_cost_nodes;
    std::map<std::string, MergedCostNode&> merged_cost_node_map;
    std::map<std::string, std::vector<Edge>> succ_edges;
    std::map<std::string, std::vector<Edge>> pred_edges;

  public:
    MergedCostGraph() {}
    MergedCostGraph(MergedCostGraph* merged_cost_graph)
      : merged_cost_nodes(merged_cost_graph->get_merged_cost_nodes()),
        merged_cost_node_map(merged_cost_graph->get_merged_cost_node_map()),
        succ_edges(merged_cost_graph->get_succ_edges()),
        pred_edges(merged_cost_graph->get_pred_edges()) {}
    virtual ~MergedCostGraph() {}
    GEN_ACCESSOR_IN_DEC(std::vector<MergedCostNode>, merged_cost_nodes)
    GEN_ACCESSOR_IN_DEC(ALL(std::map<std::string, MergedCostNode&>), merged_cost_node_map)
    GEN_ACCESSOR_IN_DEC(ALL(std::map<std::string, std::vector<Edge>>), succ_edges)
    GEN_ACCESSOR_IN_DEC(ALL(std::map<std::string, std::vector<Edge>>), pred_edges)
    void add_node(MergedCostNode node) {
      merged_cost_node_map.insert(std::pair<std::string, MergedCostNode&>(node.get_name(), node));
      merged_cost_nodes.push_back(node);
    }
    MergedCostNode& get_node(std::string name) { return merged_cost_node_map.find(name)->second; }

};
}

#endif