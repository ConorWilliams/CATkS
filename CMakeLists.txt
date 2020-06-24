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
add_subdirectory("${PROJECT_SOURCE_DIR}/external/eigen")


# bad practice
set(CMAKE_CXX_COMPILER /home/cdt1902/usr/bin/g++10)

# for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add external depandacies include paths
include_directories(SYSTEM external/smhasher/src
                           external/pcg-cpp/include
                           external/json/include)

# all code is in src
include_directories(src)



# add the executables
add_executable(bfgs src/L_BFGS.test.cpp)
target_compile_options(bfgs PRIVATE -Wall -Wextra -pedantic -Werror -O3)
target_link_libraries(bfgs PRIVATE Eigen3::Eigen)

add_executable(2d src/2d.test.cpp)
target_compile_options(2d PRIVATE -Wall -Wextra -pedantic -Werror -O3)
target_link_libraries(2d PRIVATE Eigen3::Eigen)

add_executable(quiver plt/quiver/quiver.cpp)
target_compile_options(quiver PRIVATE -Wall -Wextra -pedantic -Werror -O3)
target_link_libraries(quiver PRIVATE Eigen3::Eigen)

add_executable(sort src/sort.test.cpp)
target_compile_options(sort PRIVATE -Wall -Wextra -pedantic -O3 -march=native)
target_link_libraries(sort PRIVATE Eigen3::Eigen)

add_executable(force src/force.test.cpp)
target_compile_options(force PRIVATE -Wall -Wextra -pedantic -Werror -O3)
target_link_libraries(force PRIVATE Eigen3::Eigen)

add_executable(xmd src/vmd.test.cpp)
target_compile_options(xmd PRIVATE -Wall -Wextra -pedantic -Werror -O3 -march=native )
target_link_libraries(xmd PRIVATE Eigen3::Eigen)

add_executable(colour src/colour.test.cpp)
target_compile_options(colour PRIVATE -Wall -Wextra -pedantic -Werror -O3 -march=native )
target_link_libraries(colour PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json SMHasherSupport)







#////////////////////naughty///////////////////////

include_directories(external/nauty27r1)

add_library(Nauty SHARED IMPORTED)
set_target_properties(Nauty PROPERTIES IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/external/nauty27r1/nauty.a")

add_executable(nauty src/nauty.test.cpp)
target_link_libraries(nauty PRIVATE Nauty)
