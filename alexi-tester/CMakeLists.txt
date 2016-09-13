# -----------------
# Target: alexi-tester
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

add_executable(alexi-tester ${SRC})

target_link_libraries(
    alexi-tester
    ${Boost_LIBRARIES}
    aann
)
