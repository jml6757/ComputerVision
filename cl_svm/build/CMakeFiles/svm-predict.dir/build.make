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
include CMakeFiles/svm-predict.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/svm-predict.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/svm-predict.dir/flags.make

CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj: CMakeFiles/svm-predict.dir/flags.make
CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj: CMakeFiles/svm-predict.dir/includes_C.rsp
CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj: ../code/src/svm-predict/svm-predict.c
	$(CMAKE_COMMAND) -E cmake_progress_report C:\StereoImaging\cl_svm\build\CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj"
	C:\MinGW\bin\g++.exe  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles\svm-predict.dir\code\src\svm-predict\svm-predict.c.obj   -c C:\StereoImaging\cl_svm\code\src\svm-predict\svm-predict.c

CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.i"
	C:\MinGW\bin\g++.exe  $(C_DEFINES) $(C_FLAGS) -E C:\StereoImaging\cl_svm\code\src\svm-predict\svm-predict.c > CMakeFiles\svm-predict.dir\code\src\svm-predict\svm-predict.c.i

CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.s"
	C:\MinGW\bin\g++.exe  $(C_DEFINES) $(C_FLAGS) -S C:\StereoImaging\cl_svm\code\src\svm-predict\svm-predict.c -o CMakeFiles\svm-predict.dir\code\src\svm-predict\svm-predict.c.s

CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj.requires:
.PHONY : CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj.requires

CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj.provides: CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj.requires
	$(MAKE) -f CMakeFiles\svm-predict.dir\build.make CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj.provides.build
.PHONY : CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj.provides

CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj.provides.build: CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj

# Object files for target svm-predict
svm__predict_OBJECTS = \
"CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj"

# External object files for target svm-predict
svm__predict_EXTERNAL_OBJECTS =

svm-predict.exe: CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj
svm-predict.exe: CMakeFiles/svm-predict.dir/build.make
svm-predict.exe: libsvm_lib.a
svm-predict.exe: CMakeFiles/svm-predict.dir/objects1.rsp
svm-predict.exe: CMakeFiles/svm-predict.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable svm-predict.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\svm-predict.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/svm-predict.dir/build: svm-predict.exe
.PHONY : CMakeFiles/svm-predict.dir/build

CMakeFiles/svm-predict.dir/requires: CMakeFiles/svm-predict.dir/code/src/svm-predict/svm-predict.c.obj.requires
.PHONY : CMakeFiles/svm-predict.dir/requires

CMakeFiles/svm-predict.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\svm-predict.dir\cmake_clean.cmake
.PHONY : CMakeFiles/svm-predict.dir/clean

CMakeFiles/svm-predict.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\StereoImaging\cl_svm C:\StereoImaging\cl_svm C:\StereoImaging\cl_svm\build C:\StereoImaging\cl_svm\build C:\StereoImaging\cl_svm\build\CMakeFiles\svm-predict.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/svm-predict.dir/depend

