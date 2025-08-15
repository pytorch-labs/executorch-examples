#include <emscripten.h>
#include <emscripten/bind.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <executorch/runtime/platform/compiler.h>
#include <cstdio>

using namespace emscripten;
using tokenizers::Error;

#define THROW_JS_ERROR(errorType, message, ...)                           \
  ({                                                                      \
    char msg_buf[256];                                                    \
    int len = snprintf(msg_buf, sizeof(msg_buf), message, ##__VA_ARGS__); \
    if (len < sizeof(msg_buf)) {                                          \
      EM_ASM(throw new errorType(UTF8ToString($0)), msg_buf);             \
    } else {                                                              \
      std::string msg;                                                    \
      msg.resize(len);                                                    \
      snprintf(&msg[0], len + 1, message, ##__VA_ARGS__);                 \
      EM_ASM(throw new errorType(UTF8ToString($0)), msg.c_str());         \
    }                                                                     \
    __builtin_unreachable();                                              \
  })

/// Throws a JavaScript Error with the provided message if `error` is not `Ok`.
#define THROW_IF_ERROR(error, message, ...)          \
  ({                                                 \
    if ET_UNLIKELY ((error) != Error::Ok) {          \
      THROW_JS_ERROR(Error, message, ##__VA_ARGS__); \
    }                                                \
  })

namespace {

class Tokenizer {
 public:
  Tokenizer() : tokenizer_(std::make_unique<::tokenizers::SPTokenizer>()) {}
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&&) = default;
  Tokenizer& operator=(Tokenizer&&) = default;

  void load_from_uint8_array(val data) {
    FILE* tmp_file = fopen("/tmp/tokenizer.model", "wb");
    if (tmp_file == nullptr) {
      THROW_JS_ERROR(Error, "Failed to open file");
    }
    size_t length = data["length"].as<size_t>();
    std::vector<uint8_t> buffer(length);
    val memory_view = val(typed_memory_view(length, buffer.data()));
    memory_view.call<void>("set", data);
    fwrite(buffer.data(), sizeof(uint8_t), length, tmp_file);
    fclose(tmp_file);
    Error error = tokenizer_->load("/tmp/tokenizer.model");
    THROW_IF_ERROR(error, "Failed to load tokenizer");
    remove("/tmp/tokenizer.model");
  }

  void load(val data) {
    if (data.isString()) {
      Error error = tokenizer_->load(data.as<std::string>());
      THROW_IF_ERROR(error, "Failed to load tokenizer");
    } else if (data.instanceof (val::global("Uint8Array"))) {
      return load_from_uint8_array(data);
    } else if (data.instanceof (val::global("ArrayBuffer"))) {
      return load_from_uint8_array(val::global("Uint8Array").new_(data));
    } else {
      THROW_JS_ERROR(
          TypeError,
          "Unsupported data type: %s",
          data.typeOf().as<std::string>().c_str());
    }
  }

  val encode(const std::string& text, int8_t bos, int8_t eos) const {
    auto res = tokenizer_->encode(text, bos, eos);
    THROW_IF_ERROR(res.error(), "Failed to encode text");
    return val::array(res.get().begin(), res.get().end());
  }

  std::string decode(uint64_t prev, uint64_t current) const {
    auto res = tokenizer_->decode(prev, current);
    THROW_IF_ERROR(res.error(), "Failed to decode token");
    return res.get();
  }

  uint64_t vocab_size() const {
    return tokenizer_->vocab_size();
  }

  uint64_t bos_tok() const {
    return tokenizer_->bos_tok();
  }

  uint64_t eos_tok() const {
    return tokenizer_->eos_tok();
  }

  bool is_loaded() const {
    return tokenizer_->is_loaded();
  }

 private:
  std::unique_ptr<::tokenizers::SPTokenizer> tokenizer_;
};

} // namespace

EMSCRIPTEN_BINDINGS(TokenizerModule) {
  class_<Tokenizer>("SPTokenizer")
      .constructor<>()
      .function("load", &Tokenizer::load)
      .function("encode", &Tokenizer::encode)
      .function("decode", &Tokenizer::decode)
      .property("vocabSize", &Tokenizer::vocab_size)
      .property("bosTok", &Tokenizer::bos_tok)
      .property("eosTok", &Tokenizer::eos_tok)
      .property("isLoaded", &Tokenizer::is_loaded);
}
