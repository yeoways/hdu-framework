import networkx as nx

class Graph:
  def __init__(self):
    self.graph = nx.DiGraph()
  
  def add_node(self, name, cost, memory, out_size, device):
    self.graph.add_node(name, cost=cost, mem=memory, 
                        out_size=out_size, device=device)
          
  def add_edge(self, name_from, name_to):
    self.graph.add_edge(name_from, name_to)
  
  def number_of_nodes(self):
    return self.graph.number_of_nodes()

  def get_DiGraph(self):
    return self.graph