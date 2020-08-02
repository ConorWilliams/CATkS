cmake_minimum_required(VERSION 3.10)

# set the project name
project(CATkS VERSION 0.1
              DESCRIPTION "Continuous Adaptive kinetic (Monte Carlo) Simulator"
              LANGUAGES CXX )

# for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# cxx standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#CUDA standard
 # set(CMAKE_CUDA_STANDARD 14)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg -march=native -O1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native ")



find_package (Threads)


find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
endif()


set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

add_subdirectory("${PROJECT_SOURCE_DIR}/external/json")

# ///// EIGEN ////

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(SYSTEM external/eigen)

#////////////////////naughty///////////////////////


add_library(Nauty STATIC IMPORTED)

set_target_properties(Nauty PROPERTIES
      IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/external/nauty27r1/nauty.a"
      INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/external/nauty27r1"
)

# ////////////////////MurmerHash//////////////////////////

add_library(MHasher STATIC ${PROJECT_SOURCE_DIR}/external/smhasher/src/MurmurHash3.cpp)
target_include_directories(MHasher PUBLIC ${PROJECT_SOURCE_DIR}/external/smhasher/src)


# /////////////////PCG RANDOM///////////////////////

add_library(pcg INTERFACE)
target_include_directories(pcg INTERFACE ${PROJECT_SOURCE_DIR}/external/pcg-cpp/include)

#/////////////////////    project  ////////////////////////////////

add_executable(diffusion src/diffusion.cpp)
target_link_libraries(diffusion PRIVATE Eigen3::Eigen
                                        nlohmann_json::nlohmann_json
                                        MHasher
                                        Nauty
                                        pcg
                                        ${CMAKE_THREAD_LIBS_INIT})


target_include_directories(diffusion PRIVATE ${PROJECT_SOURCE_DIR}/src)
