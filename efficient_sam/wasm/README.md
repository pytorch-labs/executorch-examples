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

3. Generate the EfficientSAM binary file for this demo.

```bash
bash export.sh
```
It should output a file called `xnnpack_efficient_sam.pte`.

## Building and Running

Once you have Emscripten installed, ExecuTorch set up, and the model file generated, you can build and run the demo. Building may take up to 9 minutes.

```bash
cd efficient_sam/wasm # The directory containing this README

# Build the demo
bash build.sh

# Run the demo
python3 -m http.server --directory build/
```

The page will be available at http://localhost:8000/demo.html.

## Demo Features

- Load a model from a file
  - This demo only supports the EfficientSAM model. Passing in a model with different input/output shapes will result in an error.
- Run inference on an image
  - Supported formats: `.png`, `.gif`, `.jpeg`, `.jpg`
- Select a point on the image to run inference
  - May take around 6.5 seconds to run inference
- Show and hide the segmentation mask
