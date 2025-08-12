#!/bin/bash
set -e

# Required to use the presets.
cd executorch

# Create build directory if it doesn't exist
rm -rf cmake-out
cmake --preset llm \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-out -j9 --target install --config Release

echo "Building llama runner"
dir="examples/models/llama"
cmake \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -Bcmake-out/${dir} \
    ${dir}
cmake --build cmake-out/${dir} -j9 --config Release

#  cmake-out/examples/models/llama/llama_main --model_path=/data/users/lfq/executorch-examples/program-data-separation/llama_3_2_1B.pte --data_path=/data/users/lfq/executorch-examples/program-data-separation/foundation.ptd --prompt="What happens if you eat watermelon seeds?" --tokenizer_path=/data/users/lfq/hf-artifacts/tokenizer.model --temperature=0
