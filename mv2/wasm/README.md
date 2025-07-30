# ExecuTorch JavaScript Bindings Demo

This demo showcases the capabilities of ExecuTorch's JavaScript bindings. It is able to load a model, run inference, and classify an image natively in the browser.

## Prerequisites

- [Emscripten](https://emscripten.org/docs/getting_started/Tutorial.html)

## Building and Running

```
# Build the demo
bash build.sh

# Run the demo
python3 -m http.server --directory build/
```

The page will be available at http://localhost:8000/demo.html.

## Demo Features

- Load a model from a file
  - It currently only supports the MobileNetv2 model. Passing in a model with different input/output shapes will result in an error.
  - You can generate the model file by following the instructions in the [Portable Mode Readme](https://github.com/pytorch/executorch/blob/main/examples/portable/README.md).
- Run inference on an image
