cmake_minimum_required(VERSION 3.5)
project(visua VERSION 0.0.1 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCL_INCLUDE_DIRS})

# Add executable
add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} OpenCL::OpenCL nlohmann_json::nlohmann_json)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_auto_type)