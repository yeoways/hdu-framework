# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph/build

# Include any dependencies generated for this target.
include CMakeFiles/cost_graph_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cost_graph_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cost_graph_test.dir/flags.make

CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.o: CMakeFiles/cost_graph_test.dir/flags.make
CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.o: ../cost_graph_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.o -c /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph/cost_graph_test.cpp

CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph/cost_graph_test.cpp > CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.i

CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph/cost_graph_test.cpp -o CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.s

# Object files for target cost_graph_test
cost_graph_test_OBJECTS = \
"CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.o"

# External object files for target cost_graph_test
cost_graph_test_EXTERNAL_OBJECTS =

cost_graph_test: CMakeFiles/cost_graph_test.dir/cost_graph_test.cpp.o
cost_graph_test: CMakeFiles/cost_graph_test.dir/build.make
cost_graph_test: CMakeFiles/cost_graph_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cost_graph_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cost_graph_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cost_graph_test.dir/build: cost_graph_test

.PHONY : CMakeFiles/cost_graph_test.dir/build

CMakeFiles/cost_graph_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cost_graph_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cost_graph_test.dir/clean

CMakeFiles/cost_graph_test.dir/depend:
	cd /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph/build /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph/build /mnt/e/CodeHouse/hdu-framework/framework/ccsrc/cost_graph/include/cost_graph/build/CMakeFiles/cost_graph_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cost_graph_test.dir/depend

