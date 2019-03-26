CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
set(PROJECT_NAME gp_test)

add_executable(${PROJECT_NAME} test.cpp ../third_party/catch.hpp)
target_link_libraries(
    ${PROJECT_NAME}    
    PUBLIC
    gpcpp
) 

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ../third_party/
)
