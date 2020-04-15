cmake_minimum_required(VERSION 3.15)
project(cpplenet)

set(CMAKE_CXX_STANDARD 14)

add_executable(cpplenet Main_LeNet.h Main_LeNet.cpp CNN.h CNN.cpp
                Array.h Array2D.cpp Array2D.h Array3D.cpp Array3D.h
                maths_vector.h maths_vector.cpp maths_matrix.h maths_matrix.cpp
                maths_image_windows.cpp maths_image.cpp maths_image.h maths_down_sample.cpp maths_down_sample.h
                maths_convolution.h maths_convolution.cpp maths.h maths.cpp
                maths_activation_function.h maths_activation_function.cpp
                )

set(OpenCV_FOUND 1)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# linking
target_link_libraries(cpplenet ${OpenCV_LIBS})