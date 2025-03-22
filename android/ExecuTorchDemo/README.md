# ExecuTorch Android Demo App

This guide explains how to setup ExecuTorch for Android using a demo app. The app employs a [DeepLab v3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) model for image segmentation tasks. Models are exported to ExecuTorch using [XNNPACK FP32 backend](https://pytorch.org/executorch/main/backends-xnnpack.html#xnnpack-backend).

## Prerequisites
* Download and install [Android Studio and SDK 34](https://developer.android.com/studio).
* (For exporting the DL3 model) Python 3.10+ with `executorch` package installed.

## Exporting the model
Run the following python code to export the model:
```python
import torch
import torchvision.models as models
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT').eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()],
).to_executorch()

with open("dl3_xnnpack_fp32.pte", "wb") as file:
    et_program.write_to_file(file)
```

## Push the model to the phone
The app loads a hardcoded model path (`/data/local/tmp/dl3_xnnpack_fp32.pte`) on the phone.
Run the following adb command to push the model.
```
adb push dl3_xnnpack_fp32.pte /data/local/tmp/dl3_xnnpack_fp32.pte
```

## Build and install to your phone
```
./gradlew installDebug
```

## Run unit test
```
./gradlew connectedAndroidTest
```
