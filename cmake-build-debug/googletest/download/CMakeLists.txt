cmake_minimum_required(VERSION 3.10)
project(googletest LANGUAGES C CXX)

include(ExternalProject)

set (GTEST_HOME /home/goncalovirginia/CLionProjects/CP-Project/../googletest)

ExternalProject_Add(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
    SOURCE_DIR "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/src"
    BINARY_DIR "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/build"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)
