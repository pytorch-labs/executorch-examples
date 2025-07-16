
cmake_minimum_required(VERSION 3.29 FATAL_ERROR)
project(executorch_ptd CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set options for executorch build.
option(EXECUTORCH_ENABLE_LOGGING "" ON)
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON)

# Add ExecuTorch subdirectory
add_subdirectory("executorch")

set(DEMO_SOURCES main.cpp)

# Create executable
add_executable(executorch_program_data_separation ${DEMO_SOURCES})

# Include directories
target_include_directories(executorch_program_data_separation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(
  executorch_program_data_separation
  PRIVATE executorch
          extension_module_static
          extension_flat_tensor
          extension_tensor
          xnnpack_backend
          portable_ops_lib
          portable_kernels
          gflags
)

# Set output directory
set_target_properties(executorch_program_data_separation
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
