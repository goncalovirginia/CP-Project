set(TEST_TARGET "template_test")

set (TEST_SOURCES
        template_test.cpp)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES})


set(TEST_TARGET "histogram_eq_test")

set (TEST_SOURCES
        histogram_eq_test.cpp
        ../src/histogram_eq.cpp)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES} wb)