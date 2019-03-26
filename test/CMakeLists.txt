CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
set(PROJECT_NAME gp_test)
find_package(PythonLibs 2.7)

add_executable(${PROJECT_NAME} test.cpp ../third_party/catch.hpp ../third_party/matplotlibcpp.h)
target_link_libraries(
    ${PROJECT_NAME}    
    PUBLIC
    ${PYTHON_LIBRARIES}
    gpcpp
) 

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    ../third_party/
)
