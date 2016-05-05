cmake_minimum_required(VERSION 2.8)
project(alexi-ann)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3 -march=native")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    #Clang does not support OMP, so use C++ threads
    add_definitions( -DBLAZE_USE_CPP_THREADS )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -qopenmp -O3 -march=native")
endif()
message(${CMAKE_CXX_FLAGS})


# ----------
# Find Boost
# ----------

find_package(Boost COMPONENTS system filesystem program_options thread REQUIRED)
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
    fann
)
