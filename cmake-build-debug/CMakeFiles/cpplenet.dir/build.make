# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/xiao/CLionProjects/cpplenet

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/xiao/CLionProjects/cpplenet/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/cpplenet.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cpplenet.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cpplenet.dir/flags.make

CMakeFiles/cpplenet.dir/Main_LeNet.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/Main_LeNet.cpp.o: ../Main_LeNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cpplenet.dir/Main_LeNet.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/Main_LeNet.cpp.o -c /Users/xiao/CLionProjects/cpplenet/Main_LeNet.cpp

CMakeFiles/cpplenet.dir/Main_LeNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/Main_LeNet.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/Main_LeNet.cpp > CMakeFiles/cpplenet.dir/Main_LeNet.cpp.i

CMakeFiles/cpplenet.dir/Main_LeNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/Main_LeNet.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/Main_LeNet.cpp -o CMakeFiles/cpplenet.dir/Main_LeNet.cpp.s

CMakeFiles/cpplenet.dir/CNN.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/CNN.cpp.o: ../CNN.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cpplenet.dir/CNN.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/CNN.cpp.o -c /Users/xiao/CLionProjects/cpplenet/CNN.cpp

CMakeFiles/cpplenet.dir/CNN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/CNN.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/CNN.cpp > CMakeFiles/cpplenet.dir/CNN.cpp.i

CMakeFiles/cpplenet.dir/CNN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/CNN.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/CNN.cpp -o CMakeFiles/cpplenet.dir/CNN.cpp.s

CMakeFiles/cpplenet.dir/Array2D.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/Array2D.cpp.o: ../Array2D.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/cpplenet.dir/Array2D.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/Array2D.cpp.o -c /Users/xiao/CLionProjects/cpplenet/Array2D.cpp

CMakeFiles/cpplenet.dir/Array2D.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/Array2D.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/Array2D.cpp > CMakeFiles/cpplenet.dir/Array2D.cpp.i

CMakeFiles/cpplenet.dir/Array2D.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/Array2D.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/Array2D.cpp -o CMakeFiles/cpplenet.dir/Array2D.cpp.s

CMakeFiles/cpplenet.dir/Array3D.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/Array3D.cpp.o: ../Array3D.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/cpplenet.dir/Array3D.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/Array3D.cpp.o -c /Users/xiao/CLionProjects/cpplenet/Array3D.cpp

CMakeFiles/cpplenet.dir/Array3D.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/Array3D.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/Array3D.cpp > CMakeFiles/cpplenet.dir/Array3D.cpp.i

CMakeFiles/cpplenet.dir/Array3D.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/Array3D.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/Array3D.cpp -o CMakeFiles/cpplenet.dir/Array3D.cpp.s

CMakeFiles/cpplenet.dir/maths_vector.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/maths_vector.cpp.o: ../maths_vector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/cpplenet.dir/maths_vector.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/maths_vector.cpp.o -c /Users/xiao/CLionProjects/cpplenet/maths_vector.cpp

CMakeFiles/cpplenet.dir/maths_vector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/maths_vector.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/maths_vector.cpp > CMakeFiles/cpplenet.dir/maths_vector.cpp.i

CMakeFiles/cpplenet.dir/maths_vector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/maths_vector.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/maths_vector.cpp -o CMakeFiles/cpplenet.dir/maths_vector.cpp.s

CMakeFiles/cpplenet.dir/maths_matrix.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/maths_matrix.cpp.o: ../maths_matrix.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/cpplenet.dir/maths_matrix.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/maths_matrix.cpp.o -c /Users/xiao/CLionProjects/cpplenet/maths_matrix.cpp

CMakeFiles/cpplenet.dir/maths_matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/maths_matrix.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/maths_matrix.cpp > CMakeFiles/cpplenet.dir/maths_matrix.cpp.i

CMakeFiles/cpplenet.dir/maths_matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/maths_matrix.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/maths_matrix.cpp -o CMakeFiles/cpplenet.dir/maths_matrix.cpp.s

CMakeFiles/cpplenet.dir/maths_image_windows.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/maths_image_windows.cpp.o: ../maths_image_windows.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/cpplenet.dir/maths_image_windows.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/maths_image_windows.cpp.o -c /Users/xiao/CLionProjects/cpplenet/maths_image_windows.cpp

CMakeFiles/cpplenet.dir/maths_image_windows.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/maths_image_windows.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/maths_image_windows.cpp > CMakeFiles/cpplenet.dir/maths_image_windows.cpp.i

CMakeFiles/cpplenet.dir/maths_image_windows.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/maths_image_windows.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/maths_image_windows.cpp -o CMakeFiles/cpplenet.dir/maths_image_windows.cpp.s

CMakeFiles/cpplenet.dir/maths_image.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/maths_image.cpp.o: ../maths_image.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/cpplenet.dir/maths_image.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/maths_image.cpp.o -c /Users/xiao/CLionProjects/cpplenet/maths_image.cpp

CMakeFiles/cpplenet.dir/maths_image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/maths_image.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/maths_image.cpp > CMakeFiles/cpplenet.dir/maths_image.cpp.i

CMakeFiles/cpplenet.dir/maths_image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/maths_image.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/maths_image.cpp -o CMakeFiles/cpplenet.dir/maths_image.cpp.s

CMakeFiles/cpplenet.dir/maths_down_sample.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/maths_down_sample.cpp.o: ../maths_down_sample.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/cpplenet.dir/maths_down_sample.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/maths_down_sample.cpp.o -c /Users/xiao/CLionProjects/cpplenet/maths_down_sample.cpp

CMakeFiles/cpplenet.dir/maths_down_sample.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/maths_down_sample.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/maths_down_sample.cpp > CMakeFiles/cpplenet.dir/maths_down_sample.cpp.i

CMakeFiles/cpplenet.dir/maths_down_sample.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/maths_down_sample.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/maths_down_sample.cpp -o CMakeFiles/cpplenet.dir/maths_down_sample.cpp.s

CMakeFiles/cpplenet.dir/maths_convolution.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/maths_convolution.cpp.o: ../maths_convolution.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/cpplenet.dir/maths_convolution.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/maths_convolution.cpp.o -c /Users/xiao/CLionProjects/cpplenet/maths_convolution.cpp

CMakeFiles/cpplenet.dir/maths_convolution.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/maths_convolution.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/maths_convolution.cpp > CMakeFiles/cpplenet.dir/maths_convolution.cpp.i

CMakeFiles/cpplenet.dir/maths_convolution.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/maths_convolution.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/maths_convolution.cpp -o CMakeFiles/cpplenet.dir/maths_convolution.cpp.s

CMakeFiles/cpplenet.dir/maths.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/maths.cpp.o: ../maths.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/cpplenet.dir/maths.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/maths.cpp.o -c /Users/xiao/CLionProjects/cpplenet/maths.cpp

CMakeFiles/cpplenet.dir/maths.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/maths.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/maths.cpp > CMakeFiles/cpplenet.dir/maths.cpp.i

CMakeFiles/cpplenet.dir/maths.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/maths.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/maths.cpp -o CMakeFiles/cpplenet.dir/maths.cpp.s

CMakeFiles/cpplenet.dir/maths_activation_function.cpp.o: CMakeFiles/cpplenet.dir/flags.make
CMakeFiles/cpplenet.dir/maths_activation_function.cpp.o: ../maths_activation_function.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/cpplenet.dir/maths_activation_function.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpplenet.dir/maths_activation_function.cpp.o -c /Users/xiao/CLionProjects/cpplenet/maths_activation_function.cpp

CMakeFiles/cpplenet.dir/maths_activation_function.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpplenet.dir/maths_activation_function.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xiao/CLionProjects/cpplenet/maths_activation_function.cpp > CMakeFiles/cpplenet.dir/maths_activation_function.cpp.i

CMakeFiles/cpplenet.dir/maths_activation_function.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpplenet.dir/maths_activation_function.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xiao/CLionProjects/cpplenet/maths_activation_function.cpp -o CMakeFiles/cpplenet.dir/maths_activation_function.cpp.s

# Object files for target cpplenet
cpplenet_OBJECTS = \
"CMakeFiles/cpplenet.dir/Main_LeNet.cpp.o" \
"CMakeFiles/cpplenet.dir/CNN.cpp.o" \
"CMakeFiles/cpplenet.dir/Array2D.cpp.o" \
"CMakeFiles/cpplenet.dir/Array3D.cpp.o" \
"CMakeFiles/cpplenet.dir/maths_vector.cpp.o" \
"CMakeFiles/cpplenet.dir/maths_matrix.cpp.o" \
"CMakeFiles/cpplenet.dir/maths_image_windows.cpp.o" \
"CMakeFiles/cpplenet.dir/maths_image.cpp.o" \
"CMakeFiles/cpplenet.dir/maths_down_sample.cpp.o" \
"CMakeFiles/cpplenet.dir/maths_convolution.cpp.o" \
"CMakeFiles/cpplenet.dir/maths.cpp.o" \
"CMakeFiles/cpplenet.dir/maths_activation_function.cpp.o"

# External object files for target cpplenet
cpplenet_EXTERNAL_OBJECTS =

cpplenet: CMakeFiles/cpplenet.dir/Main_LeNet.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/CNN.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/Array2D.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/Array3D.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/maths_vector.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/maths_matrix.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/maths_image_windows.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/maths_image.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/maths_down_sample.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/maths_convolution.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/maths.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/maths_activation_function.cpp.o
cpplenet: CMakeFiles/cpplenet.dir/build.make
cpplenet: /usr/local/lib/libopencv_gapi.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_stitching.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_aruco.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_bgsegm.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_bioinspired.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_ccalib.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_dnn_objdetect.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_dnn_superres.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_dpm.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_face.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_freetype.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_fuzzy.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_hfs.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_img_hash.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_intensity_transform.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_line_descriptor.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_quality.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_rapid.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_reg.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_rgbd.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_saliency.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_sfm.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_stereo.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_structured_light.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_superres.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_surface_matching.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_tracking.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_videostab.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_xfeatures2d.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_xobjdetect.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_xphoto.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_highgui.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_shape.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_datasets.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_plot.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_text.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_dnn.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_ml.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_phase_unwrapping.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_optflow.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_ximgproc.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_video.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_videoio.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_imgcodecs.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_objdetect.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_calib3d.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_features2d.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_flann.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_photo.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_imgproc.4.2.0.dylib
cpplenet: /usr/local/lib/libopencv_core.4.2.0.dylib
cpplenet: CMakeFiles/cpplenet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CXX executable cpplenet"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpplenet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cpplenet.dir/build: cpplenet

.PHONY : CMakeFiles/cpplenet.dir/build

CMakeFiles/cpplenet.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cpplenet.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cpplenet.dir/clean

CMakeFiles/cpplenet.dir/depend:
	cd /Users/xiao/CLionProjects/cpplenet/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/xiao/CLionProjects/cpplenet /Users/xiao/CLionProjects/cpplenet /Users/xiao/CLionProjects/cpplenet/cmake-build-debug /Users/xiao/CLionProjects/cpplenet/cmake-build-debug /Users/xiao/CLionProjects/cpplenet/cmake-build-debug/CMakeFiles/cpplenet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cpplenet.dir/depend

