# ExecuTorch JavaScript Bindings Demo

This demo showcases the capabilities of ExecuTorch's JavaScript bindings. It is able to load a model, run inference, and classify an image natively in the browser.

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

3. Using the script `examples/portable/scripts/export.py` generate the MobileNetV2 binary file for this demo.

```bash
cd executorch # To the root of the executorch repo

# Export the model file for the demo
python3 -m examples.portable.scripts.export --model_name="mv2"
```

## Building and Running

Once you have Emscripten installed, ExecuTorch set up, and the model file generated, you can build and run the demo.

```bash
cd mv2/wasm # The directory containing this README

# Build the demo
bash build.sh

# Run the demo
python3 -m http.server --directory build/
```

The page will be available at http://localhost:8000/demo.html.

## Demo Features

- Load a model from a file
  - It currently only supports the MobileNetV2 model. Passing in a model with different input/output shapes will result in an error.
- Run inference on an image
  - Supported formats: `.png`, `.gif`, `.jpeg`, `.jpg`
