cmake_minimum_required(VERSION 2.8)
project(alexi-ann)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -qopenmp -O0 -ggdb")

# ----------
# Find Boost
# ----------

find_package(Boost 1.54 COMPONENTS system filesystem program_options thread REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# -----
# Paths
# -----

include_directories(include)
include_directories(${BLAZE_DIR})

# -----------------
# Target: alexi-ann
# -----------------

file(GLOB_RECURSE HEADERS ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS
    include/*.hpp
    include/*.h
    include/*/*.h
    include/*/*.hpp)
file(GLOB_RECURSE SRC ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS
    src/*.cpp
    src/*/*.cpp)

add_executable(alexi-ann ${SRC})

target_link_libraries(
    alexi-ann
    ${Boost_LIBRARIES}
)
