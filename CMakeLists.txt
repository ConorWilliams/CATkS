cmake_minimum_required(VERSION 3.10)

# set the project name
project(CATkS VERSION 0.1
              DESCRIPTION "Continuous Adaptive kinetic (Monte Carlo) Simulator"
              LANGUAGES CXX)

# cxx standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg -march=native -O1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

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



if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/smhasher/src/CMakeLists.txt")
    message(FATAL_ERROR "The submodules smhasher was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/pcg-cpp/README.md")
    message(FATAL_ERROR "The submodules pcg-cpp was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/json/CMakeLists.txt")
    message(FATAL_ERROR "The submodules json was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()


set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

add_subdirectory("${PROJECT_SOURCE_DIR}/external/json")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/smhasher/src")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

#threading
find_package (Threads)

# bad practice
# set(CMAKE_CXX_COMPILER /home/cdt1902/usr/bin/g++10)

# for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add external depandacies include paths
include_directories(SYSTEM external/smhasher/src
                           external/pcg-cpp/include
                           external/json/include
                           external/nauty27r1
                           external/eigen)

# all code is in src
include_directories(src)

#////////////////////naughty///////////////////////


add_library(Nauty STATIC IMPORTED)
set_target_properties(Nauty PROPERTIES IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/external/nauty27r1/nauty.a")


# //////////////////////////////////////////////

# add_executable(diffusion src/diffusion.cpp)
# target_compile_options(diffusion PRIVATE -Werror)
# target_link_libraries(diffusion PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json SMHasherSupport Nauty ${CMAKE_THREAD_LIBS_INIT})
#

add_executable(diffusion src/diffusion.cpp)
target_compile_options(diffusion PRIVATE -Werror)
target_link_libraries(diffusion PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json SMHasherSupport Nauty ${CMAKE_THREAD_LIBS_INIT})
