# ExecuTorch iOS Demo App

This app uses the [MobileNet v3](https://pytorch.org/vision/main/models/mobilenetv3.html) model to process live camera images leveraging three different backends: [XNNPACK](https://github.com/google/XNNPACK), [Core ML](https://developer.apple.com/documentation/coreml) and [Metal Performance Shaders (MPS)](https://developer.apple.com/documentation/metalperformanceshaders) (Xcode 15+ and iOS 17+ only).

<p align="center">
  <img src="https://github.com/user-attachments/assets/69f4cc2c-a95e-4e01-ba4d-7e40791716c8" width="50%">
</p>

## Prerequisites

Before we start, make sure you have the following tools installed:

### 1. Xcode 15+ and Command Line Tools

Install Xcode 15+ from the
[Mac App Store](https://apps.apple.com/app/xcode/id497799835) and then install
the Command Line Tools using the terminal:

```bash
xcode-select --install
```

### 2. Python 3.10+

Python 3.10 or above, along with `pip3`, should be pre-installed on MacOS 13.5+.
If needed, [download Python](https://www.python.org/downloads/macos/) and
install it. Verify the Python and pip versions using these commands:

```bash
which python3 pip3
python3 --version
pip3 --version
```

## Models and Labels

Now, let's move on to exporting and bundling the MobileNet v3 model.

### 1. Set Up ExecuTorch

Clone ExecuTorch and configure the basic environment:

```bash
git clone https://github.com/pytorch/executorch.git --depth 1 && cd executorch \
python3 -m venv .venv && source .venv/bin/activate && pip3 install --upgrade pip && cd - \
./executorch/install_executorch.sh
```

### 2. Install Backend Dependencies

Install additional dependencies for Core ML and MPS backends:

```bash
./executorch/backends/apple/coreml/scripts/install_requirements.sh
./executorch/backends/apple/mps/install_requirements.sh
```

### 3. Export Model

Export the MobileNet v3 model with Core ML, MPS and XNNPACK backends:

```bash
MODEL_NAME="mv3"
cd executorch && \
python3 -m examples.portable.scripts.export --model_name="$MODEL_NAME" && \
python3 -m examples.apple.coreml.scripts.export --model_name="$MODEL_NAME" && \
python3 -m examples.apple.mps.scripts.mps_example --model_name="$MODEL_NAME" && \
python3 -m examples.xnnpack.aot_compiler --model_name="$MODEL_NAME" --delegate && \
cd -
```

Move the exported model files (those with `.pte` extension) to a specific location where the Demo App will pick them up:

```bash
mkdir -p apple/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/
mv executorch/"$MODEL_NAME"*.pte apple/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/
```

### 4. Download Labels

Download the MobileNet model labels required for image classification:

```bash
curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt \
  -o apple/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/imagenet_classes.txt
```

## Final Steps

Now, we just need to open the project in Xcode, run the tests, and finally run the app.
Double-click on the project file under `apple/ExecuTorchDemo` to open it with Xcode, or run the command:

```bash
open apple/ExecuTorchDemo/ExecuTorchDemo.xcodeproj
```

### 1. Run Tests

You can run tests on Simulaltor directly in Xcode with `Cmd + U` or use the command line:

```bash
xcrun simctl create executorch "iPhone 15"
xcodebuild clean test \
     -project apple/ExecuTorchDemo/ExecuTorchDemo.xcodeproj \
     -scheme App \
     -destination name=executorch
xcrun simctl delete executorch
```

### 2. Run App

Finally, connect the device, set up Code Signing in Xcode, and then run the app using `Cmd + R`. Try installing a Release build for better performance.

Learn more about integrating and running [ExecuTorch on Apple](https://pytorch.org/executorch/main/using-executorch-ios.html) platforms.
