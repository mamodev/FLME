
file(GLOB TESTS_SRC "*.cpp")

foreach(test_file ${TESTS_SRC})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE core -luring)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
