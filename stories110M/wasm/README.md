# ExecuTorch JavaScript Bindings Demo

This demo showcases the capabilities of ExecuTorch's JavaScript bindings. It is able to load an LLM and tokenizer and generate tokens.

## Installing Emscripten

[Emscripten](https://emscripten.org/index.html) is necessary to compile ExecuTorch for Wasm. You can install Emscripten with these commands:

```bash
# Clone the emsdk repository
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk

# Download and install version 4.0.10 of the SDK
./emsdk install 4.0.10
./emsdk activate 4.0.10

# Add the Emscripten environment variables to your shell
source ./emsdk_env.sh
```

## Setting up ExecuTorch and Generating the Model File

Make sure you have the system requirements listed in the [Getting Started Guide](https://docs.pytorch.org/executorch/main/getting-started.html#system-requirements) before continuing.

1. Install ExecuTorch from PyPI.
```bash
pip3 install executorch
```

2. Update the ExecuTorch submodule.
```bash
git submodule update --init --recursive executorch
```

3. Generate the stories110M binary file and the tokenizer configuration file for this demo.

```bash
bash export.sh
```
It should output the files `stories110M.pte` and `tokenizer.model`.

## Building and Running

Once you have Emscripten installed, ExecuTorch set up, and the model and tokenizer files generated, you can build and run the demo. Building may take up to 9 minutes.

```bash
cd stories110M/wasm # The directory containing this README

# Build the demo
bash build.sh

# Run the demo
python3 -m http.server --directory build/
```

The page will be available at http://localhost:8000/demo.html.

## Demo Features

- Load a model and tokenizer configuration from a file.
  - Your browser cannot load models that are too big. The demo is configured to load the stories110M model.
  - The EOS token, BOS token, and vocab size of the tokenizer must match the model.
  - Only SentencePiece tokenizers are currently supported.
- Temperature slider ranging from 0.0 to 2.0.
- Tokens to generate slider ranging from 1 to 100.
- Generate tokens in a text box to tell a short story.
  - Display the generated tokens in a table.
  - Each token may take up to 500ms to generate depending on the length of the prompt.
