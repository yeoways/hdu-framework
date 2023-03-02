#include <cost_graph/cost_graph.hpp>
#include <iostream>
#include <cstdio>

using namespace framework;

int main() {
    CostGraph cost_graph;
    CostNode cost_node_0;
    cost_node_0.set_name("ww0");
    cost_node_0.set_compute_cost(10);
    cost_node_0.set_memory_cost(20);
    cost_node_0.set_outputs({"ww1", "ww2"});
    cost_node_0.set_output_comm_costs({20, 20});
    cost_node_0.set_device("device1");

    CostNode cost_node_1;
    cost_node_1.set_name("ww1");
    cost_node_1.set_compute_cost(20);
    cost_node_1.set_memory_cost(20);
    cost_node_1.set_inputs({"ww0"});
    cost_node_1.set_input_comm_costs({20});
    cost_node_1.set_outputs({"ww2"});
    cost_node_1.set_output_comm_costs({20});
    cost_node_1.set_device("device1");

    CostNode cost_node_2;
    cost_node_2.set_name("ww2");
    cost_node_2.set_compute_cost(30);
    cost_node_2.set_memory_cost(20);
    cost_node_2.set_inputs({"ww0", "ww1"});
    cost_node_2.set_input_comm_costs({20, 20});
    cost_node_2.set_outputs({"ww3"});
    cost_node_2.set_output_comm_costs({20});
    cost_node_2.set_device("device1");

    CostNode cost_node_3;
    cost_node_3.set_name("ww3");
    cost_node_3.set_compute_cost(40);
    cost_node_3.set_memory_cost(20);
    cost_node_3.set_inputs({"ww2"});
    cost_node_3.set_input_comm_costs({20});
    cost_node_3.set_outputs({"ww4"});
    cost_node_3.set_output_comm_costs({20});
    cost_node_3.set_device("device1");  

    CostNode cost_node_4;
    cost_node_4.set_name("ww4");
    cost_node_4.set_compute_cost(50);
    cost_node_4.set_memory_cost(20);
    cost_node_4.set_inputs({"ww3"});
    cost_node_4.set_input_comm_costs({20});
    cost_node_4.set_device("device1");

    std::vector<CostNode> cost_nodes = {cost_node_0, cost_node_1, 
        cost_node_2, cost_node_3, cost_node_4};

    cost_graph.set_cost_nodes(cost_nodes);

    for (auto cost_node : cost_graph.get_cost_nodes()) {
        std::cout << cost_node.get_name() << std::endl;
    }
}