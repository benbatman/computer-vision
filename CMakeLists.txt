cmake_minimum_required(VERSION 3.16)
project(object_detection LANGUAGES CXX)

find_package(OpenCV REQUIRED) 

add_executable(mobilenet_image src/mobilenet.cpp)
target_link_libraries(mobilenet_image PRIVATE ${OpenCV_LIBS})
target_include_directories(mobilenet_image PRIVATE ${OpenCV_INCLUDE_DIRS})
set_property(TARGET mobilenet_image PROPERTY CXX_STANDARD 23)

add_executable(mobilenet_video src/mobilenet_video.cpp)
target_link_libraries(mobilenet_video PRIVATE ${OpenCV_LIBS})
target_include_directories(mobilenet_video PRIVATE ${OpenCV_INCLUDE_DIRS})
set_property(TARGET mobilenet_video PROPERTY CXX_STANDARD 23)
