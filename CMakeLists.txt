CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
set(PROJECT_NAME gpcpp)

project(${PROJECT_NAME})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 
set(CMAKE_CXX_STANDARD 11)

INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/include.cmake)

find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} ${GP_SRC} ${GP_INCLUDES})

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${EIGEN3_INCLUDE_DIR}    
    include/
)

add_subdirectory(test)






