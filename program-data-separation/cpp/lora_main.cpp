/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <iostream>

#include <gflags/gflags.h>

DEFINE_string(lora_model_path, "llama_3_2_1B_lora.pte",
              "LoRA model serialized in flatbuffer format.");
DEFINE_string(llama_model_path, "llama_3_2_1B.pte",
              "Model serialized in flatbuffer format.")
    DEFINE_string(data_path, "foundation.ptd",
                  "Data serialized in flatbuffer format.");

using namespace ::executorch::extension;

int main(int argc, char *argv[]) {

  std::cout << "Running program-data separation example" << std::endl;
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char *lora_model_path = FLAGS_lora_model_path.c_str();
  const char *llama_model_path = FLAGS_llama_model_path.c_str();
  const char *data_path = FLAGS_data_path.c_str();

  // Load the data.
  auto data_map_loader = FileDataLoader::from(data_path);

  // Load the model.
  Module lora_module(lora_model_path, data_map_loader);
  Module llama_module(llama_model_path, data_map_loader);

  float input[3];
  auto tensor = from_blob(input, {3});

  // Perform an inference.
  const auto result = module.forward(tensor);

  if (result.ok()) {
    const auto output = result->at(0).toTensor().const_data_ptr<float>();
    for (int i = 0; i < 3; i++) {
      std::cout << output[i] << std::endl;
    }
    std::cout << "Success" << std::endl;
  }

  return 0;
}
