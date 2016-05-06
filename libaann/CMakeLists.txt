# -----------------
# Target: libaann
# -----------------

include_directories(include)

file(GLOB_RECURSE HEADERS ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS
    include/*.hpp
    include/*.h
    include/*/*.h
    include/*/*.hpp)
file(GLOB_RECURSE SRC ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS
    src/*.cpp
    src/*/*.cpp)

add_library(aann ${SRC})

target_link_libraries(
    aann
    ${Boost_LIBRARIES}
    fann
)
