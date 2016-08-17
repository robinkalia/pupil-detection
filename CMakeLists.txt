cmake_minimum_required(VERSION 2.8)
project(pupil CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/modules")

# include_directories(include)

set(source
  src/main.cpp
  src/detectIril.cpp)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pupil ${source})
target_link_libraries(pupil ${OpenCV_LIBS})
