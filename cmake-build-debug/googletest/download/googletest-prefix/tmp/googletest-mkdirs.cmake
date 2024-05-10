# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/src"
  "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/build"
  "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/download/googletest-prefix"
  "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/download/googletest-prefix/tmp"
  "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/download/googletest-prefix/src/googletest-stamp"
  "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/download/googletest-prefix/src"
  "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/download/googletest-prefix/src/googletest-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/download/googletest-prefix/src/googletest-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/goncalovirginia/CLionProjects/CP-Project/cmake-build-debug/googletest/download/googletest-prefix/src/googletest-stamp${cfgdir}") # cfgdir has leading slash
endif()
