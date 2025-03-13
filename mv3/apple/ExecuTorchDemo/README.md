# Building an ExecuTorch iOS Demo App

Welcome to the tutorial on setting up the ExecuTorch iOS Demo App!

This app uses the
[MobileNet v3](https://pytorch.org/vision/main/models/mobilenetv3.html) model to
process live camera images leveraging three different backends:
[XNNPACK](https://github.com/google/XNNPACK),
[Core ML](https://developer.apple.com/documentation/coreml) and
[Metal Performance Shaders (MPS)](https://developer.apple.com/documentation/metalperformanceshaders)
(Xcode 15+ and iOS 17+ only).

<p align="center">
  <img src="https://github.com/pytorch/executorch/blob/main/docs/source/_static/img/demo_ios_app.png" width="50%">
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

### 3. Check Swift Package Manager version
The prebuilt ExecuTorch runtime, backend, and kernels are available as a Swift PM
package. Ensure the latest SwiftPM version is linked to your Xcode installation.
The steps to add the necessary package dependency are available in the
[Using ExecuTorch for iOS](https://pytorch.org/executorch/main/using-executorch-ios.html#swift-package-manager) documentation.

### 4. Set Up ExecuTorch

Clone ExecuTorch and set up the environment as explained in the [Building from Source tutorial](https://pytorch.org/executorch/main/using-executorch-building-from-source.html):

```bash
git clone -b viable/strict https://github.com/pytorch/executorch.git && cd executorch

python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip

./install_executorch.sh
```

### 5. Backend Dependencies

Install additional dependencies for [Core ML](https://pytorch.org/executorch/main/backends-coreml.html) and [MPS](https://pytorch.org/executorch/main/backends-mps.html) backends:

```bash
./backends/apple/coreml/scripts/install_requirements.sh

./backends/apple/mps/install_requirements.sh
```

### 6. Clone the Demo App

```bash
git clone --depth 1 https://github.com/pytorch-labs/executorch-examples.git
```

## Models and Labels

Now, let's move on to exporting and bundling the MobileNet v3 model.

### 1. Export Model

Export the MobileNet v3 model using the command line and move the exported model to
a specific location where the Demo App will pick them up:

```bash
python3 -m examples.portable.scripts.export --model_name="mv3"

APP_PATH="executorch-examples/mv3/apple/ExecuTorchDemo/ExecuTorchDemo"
mkdir -p "$APP_PATH/Resources/Models/MobileNet/"
mv mv3.pte "$APP_PATH/Resources/Models/MobileNet/"
```

Next, export the MobileNet v3 model with Core ML, MPS and XNNPACK backends by runnning this [`export.py`](https://github.com/pytorch-labs/executorch-examples/mv3/apple/ExecuTorchDemo/export.py) script.

```bash
python3 export.py
```


### 2. Download Labels

Download the MobileNet model labels required for image classification:

```bash
curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt \
  -o "$APP_PATH/Resources/Models/MobileNet/imagenet_classes.txt"
```

## Final Steps

We're almost done! Now, we just need to open the project in Xcode, run the
tests, and finally run the app.

### 1. Open Project in Xcode

Double-click on the project file under
`executorch-examples/mv3/apple/ExecuTorchDemo/ExecuTorchDemo` to openit with Xcode, or run the command:

```bash
open $APP_PATH.xcodeproj
```

### 2. Run Tests

You can run tests on Simulaltor directly in Xcode with `Cmd + U` or use the
command line:

```bash
xcrun simctl create executorch "iPhone 15"
xcodebuild clean test \
     -project executorch-examples/mv3/apple/ExecuTorchDemo/ExecuTorchDemo.xcodeproj \
     -scheme App \
     -destination name=executorch
xcrun simctl delete executorch
```

### 3. Run App

Finally, connect the device, set up Code Signing in Xcode, and then run the app
using `Cmd + R`. Try installing a Release build for better performance.

Congratulations! You've successfully set up the ExecuTorch iOS Demo App. Now,
you can explore and enjoy the power of ExecuTorch on your iOS device!

Learn more about integrating and running [ExecuTorch on Apple](https://pytorch.org/executorch/main/using-executorch-ios.html) platforms.

For specific examples, take a look at this [GitHub repository](https://github.com/pytorch-labs/executorch-examples/mv3/apple/ExecuTorchDemo/).
