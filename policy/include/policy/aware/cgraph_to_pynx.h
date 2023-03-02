#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <cost_graph/cost_graph.hpp>

#ifndef _FRAMEWORK_CGRAPH_TO_PYNX_H
#define _FRAMEWORK_CGRAPH_TO_PYNX_H

using namespace framework;
namespace py = pybind11;

class ConvertCGraphToPYNX{
  public:
    CostGraph cost_graph;
    // py::scoped_interpreter python;
    py::object py_graph = py::module::import("python.py_input").attr("Graph")();

    ConvertCGraphToPYNX(){};
    ConvertCGraphToPYNX(CostGraph cost_graph_) : cost_graph(cost_graph_){};
    // ConvertCGraphToPYNX(MergedCostGraph merged_cost_graph_);

    void convert_cost_graph();
    // void convert_merged_cost_graph();
    void nx_graph_add_node(std::string name, long cost,
                           long mem, long out_size,
                           std::string device);
    void nx_graph_add_edge(std::string from, std::string to);
    int number_of_nodes();

    GEN_ACCESSOR_IN_DEC(py::object, py_graph);

};

class ConvertCMergedGraphToPYNX : public ConvertCGraphToPYNX{
  public:
    MergedCostGraph merged_cost_graph;

    ConvertCMergedGraphToPYNX() {};
    ConvertCMergedGraphToPYNX(MergedCostGraph merged_cost_graph_) 
      : merged_cost_graph(merged_cost_graph_) {};

    void convert_merged_cost_graph();
};

#endif