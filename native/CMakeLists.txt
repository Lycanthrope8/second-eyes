# Second Eyes: libse_llama (a small C interface over llama.cpp) and its test program (A1.8b, D55).
# Built by tools/build_llama.py, which fetches llama.cpp at the release in llama.cpp.pin.
cmake_minimum_required(VERSION 3.16)
project(se_llama C CXX)

set(LLAMA_DIR "" CACHE PATH "llama.cpp checkout at the release in llama.cpp.pin")
option(SE_BUILD_BENCH "also build llama.cpp's llama-bench" OFF)
if(NOT EXISTS "${LLAMA_DIR}/include/llama.h")
    message(FATAL_ERROR "LLAMA_DIR must point to a llama.cpp checkout (got '${LLAMA_DIR}')")
endif()

# llama.cpp as static libraries inside our one shared library; the options of llama.cpp's docs/android.md.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ${SE_BUILD_BENCH} CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ${SE_BUILD_BENCH} CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp.pin" SE_PIN)
string(REGEX MATCH "^([^ ]+) ([0-9a-f]+)" _ "${SE_PIN}")
string(SUBSTRING "${CMAKE_MATCH_2}" 0 8 SE_COMMIT)
set(SE_LLAMA_VERSION "${CMAKE_MATCH_1} (${SE_COMMIT})")

add_library(se_llama SHARED se_llama.cpp)
target_compile_features(se_llama PRIVATE cxx_std_17)
target_compile_definitions(se_llama PRIVATE SE_LLAMA_VERSION="${SE_LLAMA_VERSION}")
target_link_libraries(se_llama PRIVATE llama)
set_target_properties(se_llama PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)

add_executable(se_llama_cli se_llama_cli.cpp)
target_compile_features(se_llama_cli PRIVATE cxx_std_17)
target_include_directories(se_llama_cli PRIVATE "${LLAMA_DIR}/vendor")
target_link_libraries(se_llama_cli PRIVATE se_llama)
if(ANDROID)
    # found next to itself when run from a folder in adb shell (LD_LIBRARY_PATH=.)
    set_target_properties(se_llama_cli PROPERTIES BUILD_RPATH "\$ORIGIN")
endif()
