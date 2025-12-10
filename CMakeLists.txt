cmake_minimum_required(VERSION 3.10)
project(rust_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Native Linux build target ---
add_executable(rust_detector ../src/rust.cpp)

if(UNIX AND NOT WIN32)
    # Use system libcurl
    find_package(CURL REQUIRED)
    target_link_libraries(rust_detector PRIVATE CURL::libcurl)
endif()

# --- Windows cross-compile target ---
find_program(MINGW_CXX x86_64-w64-mingw32-g++)
if(MINGW_CXX)
    add_executable(rust_detector_win ../src/rust.cpp)
    set_target_properties(rust_detector_win PROPERTIES SUFFIX ".exe")

    # Point to your Windows libcurl headers/libs
    target_include_directories(rust_detector_win PRIVATE $ENV{HOME}/winlibs/curl/include)
    target_link_directories(rust_detector_win PRIVATE $ENV{HOME}/winlibs/curl/lib)

    # Link against libcurl (Windows build)
    target_link_libraries(rust_detector_win PRIVATE curl)

    # MinGW provides these stubs if your libcurl build requires them
    target_link_libraries(rust_detector_win PRIVATE ws2_32 winmm)
endif()
