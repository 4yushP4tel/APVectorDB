cmake_minimum_required(VERSION 3.26)
project(APvectorDB LANGUAGES CXX)

include(CTest)
enable_testing()

# C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# vcpkg toolchain
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Google Test
find_package(GTest CONFIG REQUIRED)

# --- DB library (shared between main and tests) ---
file(GLOB_RECURSE DB_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
list(REMOVE_ITEM DB_SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")

add_library(APvectorDB-lib STATIC ${DB_SOURCES})

target_include_directories(APvectorDB-lib PUBLIC include)

target_link_libraries(APvectorDB-lib
    PUBLIC
        Threads::Threads
)

# --- Main executable ---
add_executable(APvectorDB src/main.cpp)

target_link_libraries(APvectorDB
    PRIVATE
        APvectorDB-lib
)

set_target_properties(APvectorDB PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# --- Tests ---
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/*.cpp")

if(TEST_SOURCES)
    add_executable(APvectorDB-tests ${TEST_SOURCES})

    target_link_libraries(APvectorDB-tests
        PRIVATE
            APvectorDB-lib
            GTest::gtest
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(APvectorDB-tests)

    set_target_properties(APvectorDB-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()
