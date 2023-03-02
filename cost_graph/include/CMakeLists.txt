add_header_only_library(CostGraph
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_INSTALL_INCLUDEDIR}
    DEPENDS DistributedIRHeaders)

add_executable(cost_graph_test cost_graph_test.cpp)
target_link_libraries(cost_graph_test PUBLIC CostGraph)