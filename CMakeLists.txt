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


#
# Paths
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/lib/
   CACHE PATH
   "Single Directory for all Libraries"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/bin/
   CACHE PATH
   "Single Directory for all Executables."
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/lib/
   CACHE PATH
   "Single Directory for all static libraries."
)


# ----------
# Find Boost
# ----------

find_package(Boost COMPONENTS system filesystem program_options thread REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# -----
# Paths
# -----

include_directories(${BLAZE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/libaann/include)


# ------------
# Sub-Projects
# ------------

add_subdirectory(libaann)
add_subdirectory(alexi-trainer)
add_subdirectory(alexi-sim)
