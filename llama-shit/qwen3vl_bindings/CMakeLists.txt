cmake_minimum_required(VERSION 3.16)
project(qwen_mtmd_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

# Optional strict warning controls; disabled by default to avoid third-party header failures.
option(ENABLE_STRICT_WARNINGS "Add -Wall -Wextra -Wpedantic to compile flags" OFF)
option(ENABLE_WERROR "Treat warnings as errors (implies ENABLE_STRICT_WARNINGS)" OFF)


include(FetchContent)
find_package(Python COMPONENTS Interpreter Development QUIET)
find_package(pybind11 CONFIG QUIET)

# If not found via normal CMake paths, try to discover pybind11 from the
# active Python environment (for example when installed via pip).
if(NOT pybind11_FOUND)
  find_package(Python COMPONENTS Interpreter Development QUIET)
  if(Python_FOUND)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
      OUTPUT_VARIABLE PYBIND11_CMAKEDIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(PYBIND11_CMAKEDIR)
      message(STATUS "Found pybind11 cmake dir via Python: ${PYBIND11_CMAKEDIR}")
      set(pybind11_DIR "${PYBIND11_CMAKEDIR}")
      find_package(pybind11 CONFIG QUIET)
    endif()
  endif()
endif()

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching via FetchContent")
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Adjust paths to your llama.cpp checkout/build
set(LLAMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")
set(LLAMA_BUILD_BIN "${LLAMA_ROOT}/build/bin")

include_directories(
  ${LLAMA_ROOT}/include
  ${LLAMA_ROOT}/ggml/include
  ${LLAMA_ROOT}/tools/mtmd
  ${LLAMA_ROOT}/tools/stb
  ${LLAMA_ROOT}
)

link_directories(${LLAMA_BUILD_BIN})

pybind11_add_module(qwen_mtmd py_qwen3vl.cpp)

if(ENABLE_STRICT_WARNINGS)
  if(MSVC)
    target_compile_options(qwen_mtmd PRIVATE /W4)
  else()
    target_compile_options(qwen_mtmd PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(ENABLE_WERROR)
    if(MSVC)
      target_compile_options(qwen_mtmd PRIVATE /WX)
    else()
      target_compile_options(qwen_mtmd PRIVATE -Werror)
    endif()
  endif()
endif()

target_link_libraries(qwen_mtmd PRIVATE
  llama
  mtmd
  ggml-cuda
  ggml
  dl
  pthread
)

set_target_properties(qwen_mtmd PROPERTIES
  INSTALL_RPATH "${LLAMA_BUILD_BIN}"
)
