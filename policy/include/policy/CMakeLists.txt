
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# # aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_LIST)
# set(SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/aware/cgraph_to_pynx.cc)

# add_library (aware SHARED ${SRC_LIST})
# target_link_libraries(aware CostGraph)
# target_link_libraries(aware CommonHeaders)


add_subdirectory(aware)