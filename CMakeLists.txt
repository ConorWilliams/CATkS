cmake_minimum_required(VERSION 3.10)

# set the project name
project(CATkS VERSION 0.1
              DESCRIPTION "Continuous Adaptive kinetic (Monte Carlo) Simulator"
              LANGUAGES CXX)

# cxx standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/eigen/CMakeLists.txt")
    message(FATAL_ERROR "The submodules eigen was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/doctest/CMakeLists.txt")
    message(FATAL_ERROR "The submodules doctest was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# bad practice
set(CMAKE_CXX_COMPILER /home/cdt1902/usr/bin/g++9)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add external depandacies (header only libs)
include_directories(SYSTEM external/eigen external/doctest external/hilbert)

# all code is in src
include_directories(src)

# add the executables
add_executable(bfgs src/L_BFGS.test.cpp)
target_compile_options(bfgs PRIVATE -Wall -Wextra -pedantic -Werror -O3)

add_executable(2d src/2d.test.cpp)
target_compile_options(2d PRIVATE -Wall -Wextra -pedantic -Werror -O3)

add_executable(quiver plt/quiver/quiver.cpp)
target_compile_options(quiver PRIVATE -Wall -Wextra -pedantic -Werror -O3)

add_executable(sort src/sort.test.cpp)
target_compile_options(sort PRIVATE -Wall -Wextra -pedantic -O3 -pg)

add_executable(force src/force.test.cpp)
target_compile_options(force PRIVATE -Wall -Wextra -pedantic -Werror -O3)
