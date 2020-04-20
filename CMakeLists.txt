cmake_minimum_required(VERSION 2.8)
project(cpplenet)

set(CMAKE_CXX_STANDARD 14)

# cuda
set (CUDA-CNN_VERSION_MAJOR 2)
set (CUDA-CNN_VERSION_MINOR 0)
find_package (CUDA REQUIRED)
include_directories (${CUDA_INCLUDE_DIRS})
include_directories ("${CUDA_INCLUDE_DIRS}/../samples/common/inc")
set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -rdynamic")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_30,code=sm_30
  -gencode arch=compute_35,code=sm_35
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_50,code=compute_50
  -rdc=true
  )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  )

MESSAGE(${CUDA_NVCC_FLAGS})



#opencv
set(OpenCV_FOUND 1)
set( OpenCV_DIR /home/rcf-proj/xq2/xliang85/proj/opcv/opencv-2.4.11/release )
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



#add executable
file(GLOB SRC_FILES  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    "src/*")

cuda_add_executable(cpplenet ${SRC_FILES})


# linking
target_link_libraries(cpplenet 
	${OpenCV_LIBS}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDA_LIBRARIES}
	)
CUDA_ADD_CUBLAS_TO_TARGET(cpplenet)

