# -----------------
# Target: alexi-sim
# -----------------

include_directories(include)

# --------
# Find SDL
# --------
find_package(SDL REQUIRED)
include_directories(${SDL_INCLUDE_DIR})


file(GLOB_RECURSE HEADERS ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS
    include/*.hpp
    include/*.h
    include/*/*.h
    include/*/*.hpp)
file(GLOB_RECURSE SRC ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS
    src/*.cpp
    src/*/*.cpp)

add_executable(alexi-sim ${SRC})

target_link_libraries(
    alexi-sim
    ${Boost_LIBRARIES}
    ${SDL_LIBRARY}
    aann
)
