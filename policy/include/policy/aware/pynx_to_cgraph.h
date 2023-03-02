#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <cost_graph/cost_graph.hpp>

#ifndef _FRAMEWORK_PYNX_TO_CGRAPH_H
#define _FRAMEWORK_PYNX_TO_CGRAPH_H

using namespace framework;
namespace py = pybind11;

class ConvertPYNXToCGraph {
  public:
    CostGraph cost_graph;
    // py::scoped_interpreter python;
    py::object nx_graph = py::module::import("networkx").attr("DiGraph")();

    ConvertPYNXToCGraph(){};
    ConvertPYNXToCGraph(py::object nx_graph_) : nx_graph(nx_graph_){};
    // ConvertCGraphToPYNX(MergedCostGraph merged_cost_graph_);

    virtual void convert_py_nx_graph();
    // void convert_merged_cost_graph();
    // CostGraph get_cost_graph();
    GEN_ACCESSOR_IN_DEC(CostGraph, cost_graph);
};

class ConvertPYNXToCMergedGraph : public ConvertPYNXToCGraph {
  public:
    MergedCostGraph merged_cost_graph;

    ConvertPYNXToCMergedGraph() {};
    ConvertPYNXToCMergedGraph(py::object nx_graph_) : 
      ConvertPYNXToCGraph(nx_graph_) {};

    virtual void convert_py_nx_graph() override;
    GEN_ACCESSOR_IN_DEC(MergedCostGraph, merged_cost_graph);
};

#endif