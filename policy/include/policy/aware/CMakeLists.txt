add_subdirectory(python)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_LIST)
# set(SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/cgraph_to_pynx.cc
#              ${CMAKE_CURRENT_SOURCE_DIR}/pynx_to_cgraph.cc)

# find_package(pybind11 CONFIG REQUIRED)
# include_directories(${pybind11_INCLUDE_DIRS})
add_library(AWARE SHARED ${SRC_LIST})
target_link_libraries(AWARE CostGraph)
target_link_libraries(AWARE CommonHeaders)
target_link_libraries(AWARE pybind11::pybind11)
target_link_libraries(AWARE pybind11::embed)
# target_link_libraries(cgraph_to_pynx_test pybind11::embed)

# add_header_only_library(AWARE
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${CMAKE_INSTALL_INCLUDEDIR}
#     DEPENDS CostGraph)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/nx_graph.py DESTINATION ${CMAKE_SOURCE_DIR}/build/ccsrc/policy/include/policy/aware)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/python/py_input.py DESTINATION ${CMAKE_SOURCE_DIR}/build/ccsrc/policy/include/policy/aware/python)

add_executable(cgraph_to_pynx_test cgraph_to_pynx_test.cpp)
target_link_libraries(cgraph_to_pynx_test CostGraph)
target_link_libraries(cgraph_to_pynx_test AWARE)
# target_link_libraries(cgraph_to_pynx_test pybind11::pybind11)
# target_link_libraries(cgraph_to_pynx_test PUBLIC CostGraph)