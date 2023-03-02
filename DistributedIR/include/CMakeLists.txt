# 上一级目录
get_filename_component(ABOVE_ONE_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
# 上两级目录
get_filename_component(ABOVE_TWO_DIR ${ABOVE_ONE_DIR} DIRECTORY)
# 上三级目录
get_filename_component(ABOVE_THREE_DIR ${ABOVE_TWO_DIR} DIRECTORY)
# 上四级目录
get_filename_component(ABOVE_FOUR_DIR ${ABOVE_THREE_DIR} DIRECTORY)

# 调用add_header_only_library
include(${ABOVE_THREE_DIR}/cmake/HeaderOnlyLibrary.cmake)

add_header_only_library(DistributedIRHeaders
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_INSTALL_INCLUDEDIR}
    DEPENDS CommonHeaders)