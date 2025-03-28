cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_COMPILER clang++)

project(FLME)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_library(core STATIC)
file(GLOB_RECURSE CORE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/modules/core/*.cpp")
target_sources(core PRIVATE ${CORE_SRC})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/modules/core)

# cmds
add_executable(main_exec cmd/main.cpp)
target_link_libraries(main_exec PRIVATE core -luring)


set(Torch_DIR "${CMAKE_SOURCE_DIR}/libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
add_executable(torch_exec cmd/torch.cpp)
target_compile_options(torch_exec PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(torch_exec PRIVATE core -luring "${TORCH_LIBRARIES}")

# tests
file(GLOB TESTS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

foreach(test_file ${TESTS_SRC})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE core -luring)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

