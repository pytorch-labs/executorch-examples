#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import urllib

import torch
from executorch.runtime import Runtime
from PIL import Image
from torchvision import transforms


def get_sample_image():
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/deeplab1.png",
        "deeplab1.png",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    input_image = Image.open("deeplab1.png")
    input_image = input_image.convert("RGB")
    input_image = input_image.resize([224, 224])
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def main() -> None:
    runtime = Runtime.get()
    program = runtime.load_program("dl3_xnnpack_fp32.pte")
    method = program.load_method("forward")
    sample_image = get_sample_image()
    result = method.execute((sample_image,))[0][0]
    output_predictions = result.argmax(0)
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize([224, 224])
    r.putpalette(colors)
    r.save("result.png")


if __name__ == "__main__":
    main()
