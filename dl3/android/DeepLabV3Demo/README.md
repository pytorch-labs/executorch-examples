# ExecuTorch Android Demo App

This guide explains how to setup ExecuTorch for Android using a demo app. The app employs a [DeepLab v3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) model for image segmentation tasks. Models are exported to ExecuTorch using [XNNPACK FP32 backend](https://pytorch.org/executorch/main/backends-xnnpack.html#xnnpack-backend).

## Prerequisites
* Download and install [Android Studio and SDK 34](https://developer.android.com/studio).
* (For exporting the DL3 model) Python 3.10+ with `executorch` package installed.

## Exporting the model
Run the script in `dl3/python/export.py` to export the model.

## Push the model to the phone
The app loads a hardcoded model path (`/data/local/tmp/dl3_xnnpack_fp32.pte`) on the phone.
Run the following adb command to push the model.
```
adb push dl3_xnnpack_fp32.pte /data/local/tmp/dl3_xnnpack_fp32.pte
```

## Build and install to your phone
(`cd dl3/android/DeepLanV3Demo` first)
```
./gradlew installDebug
```

## Run unit test
```
./gradlew connectedAndroidTest
```
