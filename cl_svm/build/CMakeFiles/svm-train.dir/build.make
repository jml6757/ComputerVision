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
include CMakeFiles/svm-train.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/svm-train.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/svm-train.dir/flags.make

CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj: CMakeFiles/svm-train.dir/flags.make
CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj: CMakeFiles/svm-train.dir/includes_C.rsp
CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj: ../code/src/svm-train/svm-train.c
	$(CMAKE_COMMAND) -E cmake_progress_report C:\StereoImaging\cl_svm\build\CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj"
	C:\MinGW\bin\g++.exe  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles\svm-train.dir\code\src\svm-train\svm-train.c.obj   -c C:\StereoImaging\cl_svm\code\src\svm-train\svm-train.c

CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.i"
	C:\MinGW\bin\g++.exe  $(C_DEFINES) $(C_FLAGS) -E C:\StereoImaging\cl_svm\code\src\svm-train\svm-train.c > CMakeFiles\svm-train.dir\code\src\svm-train\svm-train.c.i

CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.s"
	C:\MinGW\bin\g++.exe  $(C_DEFINES) $(C_FLAGS) -S C:\StereoImaging\cl_svm\code\src\svm-train\svm-train.c -o CMakeFiles\svm-train.dir\code\src\svm-train\svm-train.c.s

CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj.requires:
.PHONY : CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj.requires

CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj.provides: CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj.requires
	$(MAKE) -f CMakeFiles\svm-train.dir\build.make CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj.provides.build
.PHONY : CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj.provides

CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj.provides.build: CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj

# Object files for target svm-train
svm__train_OBJECTS = \
"CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj"

# External object files for target svm-train
svm__train_EXTERNAL_OBJECTS =

svm-train.exe: CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj
svm-train.exe: CMakeFiles/svm-train.dir/build.make
svm-train.exe: libsvm_lib.a
svm-train.exe: CMakeFiles/svm-train.dir/objects1.rsp
svm-train.exe: CMakeFiles/svm-train.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable svm-train.exe"
	python stringify.py < ../code/src/opencl_kernels/linearKernelKernelSource.c > ../code/src/opencl_kernels/linearKernelKernelSource.cl
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\svm-train.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/svm-train.dir/build: svm-train.exe
.PHONY : CMakeFiles/svm-train.dir/build

CMakeFiles/svm-train.dir/requires: CMakeFiles/svm-train.dir/code/src/svm-train/svm-train.c.obj.requires
.PHONY : CMakeFiles/svm-train.dir/requires

CMakeFiles/svm-train.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\svm-train.dir\cmake_clean.cmake
.PHONY : CMakeFiles/svm-train.dir/clean

CMakeFiles/svm-train.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\StereoImaging\cl_svm C:\StereoImaging\cl_svm C:\StereoImaging\cl_svm\build C:\StereoImaging\cl_svm\build C:\StereoImaging\cl_svm\build\CMakeFiles\svm-train.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/svm-train.dir/depend
