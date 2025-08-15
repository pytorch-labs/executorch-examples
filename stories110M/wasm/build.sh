#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_OUT=build

if [ "$(uname)" == "Darwin" ]; then
    CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
    CMAKE_JOBS=$(( $(nproc) - 1 ))
fi

emcmake cmake . -DEXECUTORCH_BUILD_WASM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DFLATCC_ALLOW_WERROR=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -B"${CMAKE_OUT}"

cmake --build ${CMAKE_OUT} --target executorch_wasm_demo -j ${CMAKE_JOBS}
