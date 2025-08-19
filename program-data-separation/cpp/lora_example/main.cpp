/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */
#include <gflags/gflags.h>

#include <executorch/examples/models/llama/runner/runner.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(lora_model_path, "llama_3_2_1B_lora.pte",
              "LoRA model serialized in flatbuffer format.");
DEFINE_string(llama_model_path, "llama_3_2_1B.pte",
              "Model serialized in flatbuffer format.");
DEFINE_string(data_path, "foundation.ptd",
              "Data serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tokenizer.model", "Tokenizer stuff.");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");

DEFINE_double(temperature, 0,
              "Temperature; Default is 0. 0 = greedy argmax sampling "
              "(deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len, 128,
    "Total number of tokens to generate (prompt + output). Defaults to "
    "max_seq_len. If the number of input tokens + seq_len > max_seq_len, the "
    "output will be truncated to max_seq_len tokens.");

using namespace ::executorch::extension;

int main(int argc, char *argv[]) {
  ET_LOG(Info, "Running program-data separation lora example...");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char *lora_model_path = FLAGS_lora_model_path.c_str();
  const char *llama_model_path = FLAGS_llama_model_path.c_str();
  const char *data_path = FLAGS_data_path.c_str();

  const char *tokenizer_path = FLAGS_tokenizer_path.c_str();
  const char *prompt = FLAGS_prompt.c_str();
  float temperature = FLAGS_temperature;
  int32_t seq_len = 128;
  int32_t cpu_threads = -1;

  // Create runner for lora model.
  std::unique_ptr<::executorch::extension::llm::TextLLMRunner> lora_runner =
      example::create_llama_runner(lora_model_path, tokenizer_path, data_path);
  if (lora_runner == nullptr) {
    ET_LOG(Error, "Failed to create lora_runner.");
    return 1;
  }

  // create runner for llama model
  std::unique_ptr<::executorch::extension::llm::TextLLMRunner> llama_runner =
      example::create_llama_runner(llama_model_path, tokenizer_path, data_path);
  if (llama_runner == nullptr) {
    ET_LOG(Error, "Failed to create llama_runner.");
    return 1;
  }

  // generate
  executorch::extension::llm::GenerationConfig config{
      .seq_len = seq_len, .temperature = temperature};

  auto error = lora_runner->generate(prompt, config);
  if (error != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to generate with lora_runner, error code %zu.",
           error);
    return 1;
  }

  ET_LOG(Info, "Generating with llama...");
  error = llama_runner->generate(prompt, config);
  if (error != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to generate with llama_runner, error code %zu.",
           error);
    return 1;
  }

  return 0;
}
