# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 2.8

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files (x86)\CMake 2.8\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files (x86)\CMake 2.8\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = "C:\Program Files (x86)\CMake 2.8\bin\cmake-gui.exe"

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\StereoImaging\cl_svm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\StereoImaging\cl_svm\build

# Include any dependencies generated for this target.
include CMakeFiles/KernelTesting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/KernelTesting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/KernelTesting.dir/flags.make

CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj: CMakeFiles/KernelTesting.dir/flags.make
CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj: CMakeFiles/KernelTesting.dir/includes_CXX.rsp
CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj: ../testing/src/KernelTesting/KernelTesting.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report C:\StereoImaging\cl_svm\build\CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj"
	C:\MinGW\bin\g++.exe   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles\KernelTesting.dir\testing\src\KernelTesting\KernelTesting.cpp.obj -c C:\StereoImaging\cl_svm\testing\src\KernelTesting\KernelTesting.cpp

CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.i"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_FLAGS) -E C:\StereoImaging\cl_svm\testing\src\KernelTesting\KernelTesting.cpp > CMakeFiles\KernelTesting.dir\testing\src\KernelTesting\KernelTesting.cpp.i

CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.s"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_FLAGS) -S C:\StereoImaging\cl_svm\testing\src\KernelTesting\KernelTesting.cpp -o CMakeFiles\KernelTesting.dir\testing\src\KernelTesting\KernelTesting.cpp.s

CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj.requires:
.PHONY : CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj.requires

CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj.provides: CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj.requires
	$(MAKE) -f CMakeFiles\KernelTesting.dir\build.make CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj.provides.build
.PHONY : CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj.provides

CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj.provides.build: CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj

# Object files for target KernelTesting
KernelTesting_OBJECTS = \
"CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj"

# External object files for target KernelTesting
KernelTesting_EXTERNAL_OBJECTS =

KernelTesting.exe: CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj
KernelTesting.exe: CMakeFiles/KernelTesting.dir/build.make
KernelTesting.exe: libsvm_lib.a
KernelTesting.exe: CMakeFiles/KernelTesting.dir/objects1.rsp
KernelTesting.exe: CMakeFiles/KernelTesting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable KernelTesting.exe"
	python stringify.py < ../code/src/opencl_kernels/linearKernelKernelSource.c > ../code/src/opencl_kernels/linearKernelKernelSource.cl
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\KernelTesting.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/KernelTesting.dir/build: KernelTesting.exe
.PHONY : CMakeFiles/KernelTesting.dir/build

CMakeFiles/KernelTesting.dir/requires: CMakeFiles/KernelTesting.dir/testing/src/KernelTesting/KernelTesting.cpp.obj.requires
.PHONY : CMakeFiles/KernelTesting.dir/requires

CMakeFiles/KernelTesting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\KernelTesting.dir\cmake_clean.cmake
.PHONY : CMakeFiles/KernelTesting.dir/clean

CMakeFiles/KernelTesting.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\StereoImaging\cl_svm C:\StereoImaging\cl_svm C:\StereoImaging\cl_svm\build C:\StereoImaging\cl_svm\build C:\StereoImaging\cl_svm\build\CMakeFiles\KernelTesting.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/KernelTesting.dir/depend

