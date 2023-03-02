#include "framework/ccsrc/cost_graph/include/cost_graph/cost_graph.hpp"

using namespace framework;

class AwareFusion {
  private:
    Graph graph;
    int threshold;
    CostGraph cost_graph;
    std::vector<MergedCostNode&> merged_cost_node_vector;
    std::map<std::string, MergedCostNode&> merged_cost_node_map;

  public:
    AwareFusion () {}
    AwareFusion (Graph graph, int threshold) {}
    GEN_ACCESSOR_IN_DEC(Graph, graph)
    GEN_ACCESSOR_IN_DEC(int, threshold)
    GEN_ACCESSOR_IN_DEC(std::vector<MergedCostNode&>, merged_cost_node_vector)
    GEN_ACCESSOR_IN_DEC(ALL(std::map<std::string, MergedCostNode&>), merged_cost_node_map)

    void GenerateFusedGraph () {}
    void OperatorFusing (std::vector<MergedCostNode&>& merged_cost_node_vec, 
      std::map<std::string, MergedCostNode&>& merged_cost_nodes) {}
    void UpdateNodeInfo (std::vector<MergedCostNode&>& merged_cost_node_vec, 
      std::map<std::string, MergedCostNode&>& merged_cost_nodes, MergedCostNode& node) {}
    
    
};