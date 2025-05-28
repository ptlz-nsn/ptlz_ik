cmake_minimum_required(VERSION 3.10)
project(Test)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
set(MNN_DIR /home/host/code/python/MNN)     
include_directories(${MNN_DIR}/include)
include_directories(${MNN_DIR}/include/MNN)
include_directories(${MNN_DIR}/tools)
include_directories(${MNN_DIR}/tools/cpp)
include_directories(${MNN_DIR}/source)
include_directories(${MNN_DIR}/source/backend)
include_directories(${MNN_DIR}/source/core)

LINK_DIRECTORIES(${MNN_DIR}/build)
add_executable(main src/main.cpp)
target_link_libraries(main -lMNN ${OpenCV_LIBS})