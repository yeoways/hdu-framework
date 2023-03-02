#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <iostream>


namespace py = pybind11;


int main() {
  py::scoped_interpreter python;
  // py::module sys = py::module::import("sys");
  // py::module os = py::module::import("os");
  // py::module dirname = os.attr("path").attr("dirname");
  // py::object path = dirname(dirname(__file__));
  // sys_append(path);

  py::module c2p = py::module::import("c2p_test");

  py::object Graph = c2p.attr("Graph"); // class
  py::object graph = Graph(); // class object
  py::object add_node = graph.attr("add_node");// object method
  add_node("1", 10, 20, 20, "device1");
  py::object nx_graph = graph.attr("graph");
  py::object number_of_nodes = graph.attr("number_of_nodes");
  py::object node_num = number_of_nodes();
  py::object nxi_graph = py::module::import("networkx").attr("DiGraph")();
  nxi_graph.attr("add_node")("1");
  std::cout << nxi_graph.attr("number_of_nodes")().cast<int>();
  int node_n = node_num.cast<int>();

  std::cout << node_n << std::endl;
}